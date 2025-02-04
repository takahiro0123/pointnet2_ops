#pragma once
#include <torch/extension.h>
#include <tuple>

at::Tensor gather_points(at::Tensor points, at::Tensor idx);
at::Tensor gather_points_grad(at::Tensor grad_out, at::Tensor idx, const int n);
std::tuple<at::Tensor, at::Tensor> furthest_point_sampling(at::Tensor points, at::Tensor class_labels, const int nsamples);