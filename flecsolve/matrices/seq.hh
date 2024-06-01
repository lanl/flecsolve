/*
© 2025. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract
89233218CNA000001 for Los Alamos National Laboratory (LANL), which is
operated by Triad National Security, LLC for the U.S. Department of
Energy/National Nuclear Security Administration. All rights in the
program are reserved by Triad National Security, LLC, and the U.S.
Department of Energy/National Nuclear Security Administration. The
Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material
to reproduce, prepare. derivative works, distribute copies to the
public, perform publicly and display publicly, and to permit others
to do so.
*/
#ifndef FLECSOLVE_MATRICES_CSR_HH
#define FLECSOLVE_MATRICES_CSR_HH

#include <cstddef>
#include <limits>
#include <vector>
#include <numeric>
#include <algorithm>
#include <fstream>

#include <flecsi/util/serialize.hh>
#include <flecsi/util/array_ref.hh>

#include "flecsolve/util/traits.hh"
#include "flecsolve/vectors/seq.hh"
#include "flecsolve/operators/core.hh"

namespace flecsolve::mat {

template<template<class> class Data, template<class> class Ops, class Config>
struct sparse : op::base<> {
	using config = Config;
	using scalar = typename config::scalar;
	using size = typename config::size;
	using data_t = Data<Config>;
	using ops = Ops<Data<Config>>;

	sparse() {}
	sparse(data_t && d) : data{std::move(d)} {}

	template<class D,
	         class R,
	         std::enable_if_t<is_vector_v<D>, bool> = true,
	         std::enable_if_t<is_vector_v<R>, bool> = true>
	constexpr void apply(const D & x, R & y) const {
		return mult(x, y);
	}

	template<class X,
	         class Y,
	         std::enable_if_t<is_vector_v<X>, bool> = true,
	         std::enable_if_t<is_vector_v<Y>, bool> = true>
	constexpr void mult(const X & x, Y & y) const {
		return ops::spmv(x, data, y);
	}

	data_t data;
};

template<class Config>
struct compressed_vector_data {
	using config = Config;
	using scalar = typename config::scalar;
	using size = typename config::size;
	template<class T>
	using span = flecsi::util::span<T>;
	static constexpr bool is_resizable = true;

	void resize_major(size s) {
		offsets_.resize(s + 1);
		offsets_[0] = 0;
	}
	void resize_nnz(size s) {
		indices_.resize(s);
		values_.resize(s);
		offsets_[major_size()] = s;
	}

	constexpr span<size> offsets() {
		return {offsets_.data(), offsets_.size()};
	}
	constexpr span<const size> offsets() const {
		return {offsets_.data(), offsets_.size()};
	}
	constexpr span<size> indices() {
		return {indices_.data(), indices_.size()};
	}
	constexpr span<const size> indices() const {
		return {indices_.data(), indices_.size()};
	}
	constexpr span<scalar> values() { return {values_.data(), values_.size()}; }
	constexpr span<const scalar> values() const {
		return {values_.data(), values_.size()};
	}

	constexpr size major_size() const { return offsets_.size() - 1; }

	constexpr auto & offsets_vec() { return offsets_; }
	constexpr const auto & offsets_vec() const { return offsets_; }
	constexpr auto & indices_vec() { return indices_; }
	constexpr const auto & indices_vec() const { return indices_; }
	constexpr auto & values_vec() { return values_; }
	constexpr const auto & values_vec() const { return values_; }

	constexpr auto vecs() {
		return std::forward_as_tuple(
			offsets_vec(), indices_vec(), values_vec());
	}

	constexpr auto vecs() const {
		return std::forward_as_tuple(
			offsets_vec(), indices_vec(), values_vec());
	}

protected:
	std::vector<size> offsets_;
	std::vector<size> indices_;
	std::vector<scalar> values_;
};

template<class ScalarElement, class SizeElement>
struct compressed_view_data {
	template<class Config>
	struct type {
		using scalar_element = ScalarElement;
		using size_element = SizeElement;
		using config = Config;
		using scalar = typename config::scalar;
		using size = typename config::size;
		template<class T>
		using span = flecsi::util::span<T>;
		static constexpr bool is_resizable = false;

		type(size msize,
		     size nnz,
		     size_element * offsets,
		     size_element * indices,
		     scalar_element * values)
			: major_size_{msize}, nnz_{nnz}, offsets_{offsets},
			  indices_{indices}, values_{values} {}

		constexpr span<size_element> offsets() const {
			return {offsets_, major_size_ + 1};
		}
		constexpr span<size_element> indices() const {
			return {indices_, nnz_};
		}
		constexpr span<scalar_element> values() const {
			return {values_, nnz_};
		}

		constexpr size major_size() const { return major_size_; }
		constexpr size nnz() const { return nnz_; }

	protected:
		size major_size_;
		size nnz_;
		size_element * offsets_;
		size_element * indices_;
		scalar_element * values_;
	};
};

enum class major { col, row };

template<class Data>
struct compressed_ops {
	using scalar = typename Data::scalar;
	using size = typename Data::size;
	using config = typename Data::config;

	template<class X, class Y>
	static void spmv(const X & x, const Data & data, Y & y) {
		if constexpr (config::format == major::row) {
			const size * rowptr = data.offsets().data();
			const size * colind = data.indices().data();
			const scalar * values = data.values().data();
			for (std::size_t i = 0; i < data.major_size(); ++i) {
				y[i] = 0.;
				for (std::size_t off = rowptr[i]; off < rowptr[i + 1]; ++off) {
					y[i] += values[off] * x[colind[off]];
				}
			}
		}
		else {
			flog(error) << "Not implemented" << std::endl;
		}
	}
};

template<class Scalar, class Size, major Format>
struct compressed_config {
	static constexpr major format = Format;
	using scalar = Scalar;
	using size = Size;
};
using compressed_defaults = compressed_config<double, std::size_t, major::row>;

template<class Config = compressed_defaults,
         template<class> class Data = compressed_vector_data,
         template<class> class Ops = compressed_ops>
struct compressed : sparse<Data, Ops, Config> {
	static constexpr bool is_row_major = Config::format == major::row;
	using base = sparse<Data, Ops, Config>;
	using base::data;
	using data_t = typename base::data_t;
	using size = typename Config::size;
	using scalar = typename Config::scalar;

	template<class T>
	using span = flecsi::util::span<T>;

	compressed() : major_size_{0}, minor_size_{0}, nnz_{0} {}
	compressed(size rows, size cols, data_t d = data_t())
		: base{std::move(d)}, major_size_{is_row_major ? rows : cols},
		  minor_size_{is_row_major ? cols : rows}, nnz_{data.indices().size()} {
		if constexpr (data_t::is_resizable)
			data.resize_major(major_size_);
	}

	constexpr size rows() const {
		return is_row_major ? major_size_ : minor_size_;
	}
	constexpr size cols() const {
		return is_row_major ? minor_size_ : major_size_;
	}
	constexpr size nnz() const { return nnz_; }

	constexpr auto rep() {
		return std::make_tuple(data.offsets(), data.indices(), data.values());
	}
	constexpr auto rep() const {
		return std::make_tuple(data.offsets(), data.indices(), data.values());
	}

	void resize(size nnz) {
		static_assert(data_t::is_resizable);
		data.resize_nnz(nnz);
		nnz_ = nnz;
	}

	constexpr void set_nnz(size nnz) { nnz_ = nnz; }
	constexpr size & major_size() { return major_size_; }
	constexpr size & minor_size() { return minor_size_; }

protected:
	size major_size_;
	size minor_size_;
	size nnz_;
};

template<class scalar,
         class size = std::size_t,
         template<class> class Data = compressed_vector_data,
         template<class> class Ops = compressed_ops>
using csr = compressed<compressed_config<scalar, size, major::row>, Data, Ops>;

template<class R, class C, class V>
struct csr_view
	: compressed<compressed_config<typename V::value_type,
                                   typename R::value_type,
                                   major::row>,
                 compressed_view_data<typename V::element_type,
                                      typename R::element_type>::template type,
                 compressed_ops> {
	using base = compressed<
		compressed_config<typename V::value_type,
	                      typename R::value_type,
	                      major::row>,
		compressed_view_data<typename V::element_type,
	                         typename R::element_type>::template type,
		compressed_ops>;
	using data_t = typename base::data_t;

	constexpr csr_view(R r, C c, V v)
		: base{r.size() - 1,
	           r.size() - 1,
	           data_t{r.size() - 1, v.size(), r.data(), c.data(), v.data()}} {}
};

template<class Scalar, class Size>
struct coo_config {
	using scalar = Scalar;
	using size = Size;
};

template<class Config>
struct coo_data {
	using size = typename Config::size;
	using scalar = typename Config::scalar;

	std::vector<size> I, J;
	std::vector<scalar> V;
};

template<class Data>
struct not_implemented {
	template<class X, class Y>
	static constexpr void spmv(const X &, const Data &, Y &) {
		flog(error) << "SPMV not implemented" << std::endl;
	}
};

template<class scalar, class size = std::size_t>
struct coo : sparse<coo_data, not_implemented, coo_config<scalar, size>> {
	using base = sparse<coo_data, not_implemented, coo_config<scalar, size>>;
	using base::data;

	coo(size n, size m) : rows_{n}, cols_{m} {}

	void resize(size nnz) {
		[=](auto &... v) { ((v.resize(nnz)), ...); }(data.I, data.J, data.V);
	}

	constexpr size nnz() const { return data.V.size(); }
	constexpr size rows() const { return rows_; }
	constexpr size cols() const { return cols_; }

	auto tocsr() const {
		const auto & I = data.I;
		const auto & J = data.J;
		const auto & V = data.V;

		// argsort by rows
		std::vector<std::size_t> ind(I.size());
		std::iota(ind.begin(), ind.end(), 0);
		std::stable_sort(
			ind.begin(), ind.end(), [&](std::size_t i1, std::size_t i2) {
				return I[i1] < I[i2];
			});

		csr<scalar, size> ret{rows_, cols_};
		ret.resize(nnz());
		auto [rowptr, colind, values] = ret.rep();
		rowptr[rows_] = 0;
		for (size i = 0; i < nnz(); ++i) {
			++rowptr[I[ind[i]] + 1];
			colind[i] = J[ind[i]];
			values[i] = V[ind[i]];
		}

		for (size i = 0; i < rows(); ++i) {
			rowptr[i + 1] += rowptr[i];
		}

		return ret;
	}

protected:
	size rows_;
	size cols_;
};

}

namespace flecsi::util::serial {

template<class Config>
struct traits<flecsolve::mat::compressed_vector_data<Config>> {
	using type = flecsolve::mat::compressed_vector_data<Config>;
	using size = typename Config::size;
	using scalar = typename Config::scalar;

	template<class P>
	static void put(P & p, const type & c) {
		serial::put(p, c.offsets_vec(), c.indices_vec(), c.values_vec());
	}

	static type get(const std::byte *& p) {
		type ret;

		ret.offsets_vec() = serial::get<std::vector<size>>(p);
		ret.indices_vec() = serial::get<std::vector<size>>(p);
		ret.values_vec() = serial::get<std::vector<scalar>>(p);

		return ret;
	}
};

template<class Config, template<class> class Data, template<class> class Ops>
struct traits<flecsolve::mat::compressed<Config, Data, Ops>> {
	using type = flecsolve::mat::compressed<Config, Data, Ops>;
	using size = typename Config::size;
	using scalar = typename Config::scalar;
	using data = Data<Config>;

	template<class P>
	static void put(P & p, const type & c) {
		serial::put(p, c.rows(), c.cols(), c.data);
	}

	static type get(const std::byte *& p) {
		size rows = serial::get<size>(p);
		size cols = serial::get<size>(p);
		data d = serial::get<data>(p);
		type ret{rows, cols, std::move(d)};

		return ret;
	}
};
}

#endif
