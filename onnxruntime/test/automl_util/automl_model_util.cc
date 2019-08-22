// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <memory>
#include <google/protobuf/stubs/common.h>
#include <google/protobuf/util/delimited_message_util.h>
#include "core/graph/model.h"
#include "core/session/environment.h"
#include "onnx/defs/data_type_utils.h"
#include "automl.pb.h"

// We need to so users can specify
// path names in UNICODE
#ifndef ORT_TSTR
#ifdef _WIN32
#define ORT_TSTR(X) L##X
using ORT_CHAR = wchar_t;
using ort_string = std::wstring;
#else
#define ORT_TSTR(X) (X)
using ORT_CHAR = char;
using ort_string = std::string;
#endif
#endif

namespace onnxruntime {
namespace automl {

//!\brief Command virtual interface
struct Command {
  Command() = default;
  virtual ~Command() = default;
  Command(const Command&) = delete;
  Command& operator=(const Command&) = delete;
  // Display a portion of help for this command to
  // the specified stream.
  virtual std::ostream& Help(std::ostream&) = 0;
  //!\brief Runs the command
  // @param argc argument count
  // @param array of program arguments
  virtual void Run(int argc, const ORT_CHAR* argv[]) = 0;
};

static std::map<ort_string, std::unique_ptr<Command>> s_commands;

class HelpCommand : public Command {
 public:
  HelpCommand() = default;
  ~HelpCommand() = default;
  std::ostream& Help(std::ostream& os) override {
    return os << "\thelp - displays this help message";
  }
  void Run(int, const ORT_CHAR* []) override {
    std::ostream& os = std::cout;
    os << "A utility to create and run AutoML models\n"
          "Usage: onnxruntime_automl_util command [command specific options]";
    for (const auto& e : s_commands) {
      e.second->Help(os) << '\n';
    }
    os << std::endl;
  }
};

//!\brief Command to produce an .onnx model file
// Given a spec file
// Format of the spec file as follows:
// # Comment line recognized at the start
// Node_name:input1_name=type,input2_name=type,...:output1_name=type,...
class MakeModel : public Command {
  using InputOutput = std::pair<std::string, std::string>;

  struct NodeSpec {
    std::string name_;
    std::vector<InputOutput> inputs_;
    std::vector<InputOutput> outputs_;
  };

  InputOutput GetSpec(const std::string& spec) const {
    auto eq = spec.find('=');
    if (eq == std::string::npos) {
      ORT_THROW("No equal");
    }
    auto name = spec.substr(0, eq);
    auto type = spec.substr(eq + 1);
    if (name.empty() || type.empty()) {
      ORT_THROW("Bad InputOutput spec");
    }
    InputOutput result(std::move(name), std::move(type));
    return result;
  }

  std::vector<InputOutput> GetInputsOutputs(const std::string& spec) const {
    std::vector<InputOutput> result;
    size_t start = 0;
    auto pos = spec.find(',');
    while (pos != std::string::npos) {
      result.push_back(GetSpec(spec.substr(start, pos - start)));
      start = pos + 1;
      pos = spec.find(',', start);
    }
    auto tail = spec.substr(pos);
    if (!tail.empty()) {
      result.push_back(GetSpec(tail));
    }
    return result;
  }

  std::vector<NodeSpec> ParseSpec(const ort_string& spec_file) const {
    std::ifstream is(spec_file);
    if (!is) {
      ORT_THROW("Unable to open spec_file");
    }
    std::vector<NodeSpec> result;
    std::string line;
    size_t pos = 0;
    while (!is.eof() && std::getline(is, line)) {
      if (line.empty() || line[0] == '#') {
        continue;
      }
      pos = line.find(':', pos);
      if (pos == std::string::npos) {
        ORT_THROW("Bad spec line: ", line);
      }
      auto name = line.substr(0, pos);
      auto first_pos = pos + 1;
      pos = line.find(':', first_pos);
      if (pos == std::string::npos) {
        ORT_THROW("Bad spec line: ", line);
      }
      auto inputs = line.substr(first_pos, pos - first_pos);
      auto outputs = line.substr(pos);
      if (inputs.empty() || outputs.empty()) {
        if (pos == std::string::npos) {
          ORT_THROW("Empty inputs or outputs: ", line);
        }
      }
      auto input_specs = GetInputsOutputs(inputs);
      auto output_specs = GetInputsOutputs(outputs);
      result.push_back({std::move(name), std::move(input_specs), std::move(output_specs)});
    }
    return result;
  }

  std::string GenerateModel(const std::vector<NodeSpec>& specs) {
    using namespace ONNX_NAMESPACE;
    using namespace ONNX_NAMESPACE::Utils;
    Model model("AutoMLModel", false);
    auto& graph = model.MainGraph();

    for (const auto& spec : specs) {
      std::vector<onnxruntime::NodeArg*> inputs;
      std::vector<onnxruntime::NodeArg*> outputs;
      for (const auto& in : spec.inputs_) {
        const TypeProto& proto = DataTypeUtils::ToTypeProto(&in.second);
        auto& node_arg = graph.GetOrCreateNodeArg(in.first, &proto);
        inputs.push_back(&node_arg);
      }
      for (const auto& out : spec.outputs_) {
        const TypeProto& proto = DataTypeUtils::ToTypeProto(&out.second);
        auto& node_arg = graph.GetOrCreateNodeArg(out.first, &proto);
        outputs.push_back(&node_arg);
      }
      auto& node = graph.AddNode(spec.name_, spec.name_, "",
                                 inputs, outputs, nullptr, onnxruntime::kMSAutoMLDomain);
      node.SetExecutionProviderType(onnxruntime::kCpuExecutionProvider);
    }
    auto st = graph.Resolve();
    if (!st.IsOK()) {
      ORT_THROW("Graph Resolve:", st.ToString());
    }
    std::string serialized_model;
    auto model_proto = model.ToProto();
    if (!model_proto.SerializeToString(&serialized_model)) {
      ORT_THROW("Model serialize failed");
    }
    return serialized_model;
  }

 public:
  MakeModel() = default;
  ~MakeModel() = default;

  std::ostream& Help(std::ostream& os) override {
    return os << "\tmkmodel make onnx model. Options:\n"
                 "\t\t-i <spec file>\n"
                 "\t\t-o <output file>";
  }

  void Run(int argc, const ORT_CHAR* argv[]) override {
    if (argc < 6) {
      throw std::invalid_argument("mkmodel: not enough arguments");
    }
    ort_string in(ORT_TSTR("-i"));
    ort_string out(ORT_TSTR("-o"));
    ort_string spec_file;
    ort_string out_file;
    for (int i = 2; i < argc; ++i) {
      if (argv[i] == in && ++i < argc) {
        spec_file = argv[i];
      } else if (argv[i] == out && ++i < argc) {
        out_file = argv[i];
      }
    }
    if (spec_file.empty() || out_file.empty()) {
      throw std::invalid_argument("can not find file specs");
    }
    auto node_specs = ParseSpec(spec_file);
    auto model_str = GenerateModel(node_specs);
  }
};

}  // namespace automl
}  // namespace onnxruntime

using namespace onnxruntime;
using namespace onnxruntime::automl;

#ifdef _WIN32
int wmain(int argc, const wchar_t* argv[]) {
#else
int main(int argc, const char* argv[]) {
#endif
  std::unique_ptr<Environment> env;
  int retval = EXIT_SUCCESS;
  try {
    auto status = Environment::Create(env);
    if (!status.IsOK()) {
      ORT_THROW("Failed to create env: ", status.ToString());
    }

    s_commands.emplace(ORT_TSTR("help"), new HelpCommand);

    if (argc < 2) {
      throw std::invalid_argument("Not enough args");
    }

    auto cmd = s_commands.find(argv[1]);
    if (cmd != s_commands.end()) {
      cmd->second->Run(argc, argv);
    } else {
      throw std::invalid_argument("");
    }

  } catch (const std::invalid_argument& ex) {
    std::cerr << ex.what() << std::endl;
    s_commands.find(ORT_TSTR("help"))->second->Run(argc, argv);
    retval = EXIT_FAILURE;
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << std::endl;
    retval = EXIT_FAILURE;
  }
  // Release the protobuf library if we failed to create an env (the env will release it automatically on destruction)
  if (!env) {
    ::google::protobuf::ShutdownProtobufLibrary();
  }
  return retval;
}
