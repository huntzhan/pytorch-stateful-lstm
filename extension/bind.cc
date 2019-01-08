#include "extension/unidirectional_lstm.h"
#include "extension/stateful_unidirectional_lstm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  torch::python::bind_module<cnt::UnidirectionalSingleLayerLstm>(
      m, "UnidirectionalSingleLayerLstm")

      .def(
          py::init<
              int64_t, int64_t, int64_t,
              bool, int64_t,
              double, double,
              int64_t, double>(),
          // Required.
          py::arg("input_size"),
          py::arg("hidden_size"),
          py::arg("cell_size"),
          // Optional.
          py::arg("go_forward") = true,
          py::arg("truncated_bptt") = 0,
          py::arg("cell_clip") = 0.0,
          py::arg("proj_clip") = 0.0,
          py::arg("recurrent_dropout_type") = 0,
          py::arg("recurrent_dropout_probability") = 0.0)

      .def(
          "__call__",
          (
              cnt::LstmForwardRetType
              (cnt::UnidirectionalSingleLayerLstm::*)
              (torch::Tensor, const std::vector<int> &, cnt::LstmStateType)
          )
              &cnt::UnidirectionalSingleLayerLstm::forward)

      .def(
          "__call__",
          (
              cnt::LstmForwardRetType
              (cnt::UnidirectionalSingleLayerLstm::*)
              (torch::Tensor, const std::vector<int> &)
          )
              &cnt::UnidirectionalSingleLayerLstm::forward);

  torch::python::bind_module<cnt::UnidirectionalLstm>(
      m, "UnidirectionalLstm")

      .def(
          py::init<
              int64_t,
              int64_t, int64_t, int64_t,
              bool, int64_t,
              double, double,
              int64_t, double>(),
          // Required.
          py::arg("num_layers"),
          py::arg("input_size"),
          py::arg("hidden_size"),
          py::arg("cell_size"),
          // Optional.
          py::arg("go_forward") = true,
          py::arg("truncated_bptt") = 0,
          py::arg("cell_clip") = 0.0,
          py::arg("proj_clip") = 0.0,
          py::arg("recurrent_dropout_type") = 0,
          py::arg("recurrent_dropout_probability") = 0.0)

      .def(
          "__call__",
          (
              cnt::LstmForwardRetType
              (cnt::UnidirectionalLstm::*)
              (torch::Tensor, const std::vector<int> &, cnt::LstmStateType)
          )
              &cnt::UnidirectionalLstm::forward)

      .def(
          "__call__",
          (
              cnt::LstmForwardRetType
              (cnt::UnidirectionalLstm::*)
              (torch::Tensor, const std::vector<int> &)
          )
              &cnt::UnidirectionalLstm::forward);

  torch::python::bind_module<cnt::StatefulUnidirectionalLstm>(
      m, "StatefulUnidirectionalLstm")

      .def(
          py::init<
              int64_t,
              int64_t, int64_t, int64_t,
              bool, int64_t,
              double, double,
              int64_t, double>(),
          // Required.
          py::arg("num_layers"),
          py::arg("input_size"),
          py::arg("hidden_size"),
          py::arg("cell_size"),
          // Optional.
          py::arg("go_forward") = true,
          py::arg("truncated_bptt") = 0,
          py::arg("cell_clip") = 0.0,
          py::arg("proj_clip") = 0.0,
          py::arg("recurrent_dropout_type") = 0,
          py::arg("recurrent_dropout_probability") = 0.0)

      .def(
          "__call__",
          &cnt::StatefulUnidirectionalLstm::forward)

      .def(
          "reset_states",
          &cnt::StatefulUnidirectionalLstm::reset_states);
}
