#ifndef FLECSOLVE_MATRICES_CSR_HH
#define FLECSOLVE_MATRICES_CSR_HH

#include <limits>
#include <vector>
#include <numeric>
#include <algorithm>
#include <fstream>

#include <flecsi/util/serialize.hh>
#include <flecsi/util/array_ref.hh>

#include "flecsolve/util/traits.hh"
#include "flecsolve/vectors/seq.hh"
#include "flecsolve/operators/base.hh"

namespace flecsolve::mat {

template<class T>
struct traits {};


template<class Derived>
struct sparse : with_derived<Derived>, op::base<Derived> {
	using scalar = typename traits<Derived>::scalar_t;
	using size = typename traits<Derived>::size_t;
	using data_t = typename traits<Derived>::data_t;
	using ops_t = typename traits<Derived>::ops_t;
	using with_derived<Derived>::derived;

	sparse() {}
	sparse(data_t && d) : data{std::move(d)} {}

	constexpr size rows() const { return derived().rows(); }
	constexpr size cols() const { return derived().cols(); }
	constexpr size nnz() const { return derived().nnz(); }

	template<class D, class R>
	constexpr void apply(const vec::base<D> & x, vec::base<R> & y) const {
		return mult(x.derived(), y.derived());
	}

	template<class X, class Y>
	constexpr void mult(const vec::base<X> & x, vec::base<Y> & y) const {
		return ops.spmv(y.derived(), data, x.derived());
	}

	constexpr void resize(size nnz) {
		return derived().resize(nnz);
	}

	data_t data;
	ops_t ops;
};

template<class Scalar, class Size>
struct compressed_vector_data {
	using scalar = Scalar;
	using size = Size;
	template<class T>
	using span = flecsi::util::span<T>;
	static constexpr bool is_resizable = true;

	void resize_major(size s) {
		offsets_.resize(s+1);
		offsets_[0] = 0;
	}
	void resize_nnz(size s) {
		indices_.resize(s);
		values_.resize(s);
		offsets_[major_size()] = s;
	}

	constexpr size * offsets_ptr() { return offsets_.data(); }
	constexpr const size * offsets_ptr() const { return offsets_.data(); }

	constexpr size * indices_ptr() { return indices_.data(); }
	constexpr const size * indices_ptr() const { return indices_.data(); }

	constexpr scalar * values_ptr() { return values_.data(); }
	constexpr const scalar * values_ptr() const { return values_.data(); }


	constexpr span<size> offsets_span() { return {offsets_.data(), offsets_.size()}; }
	constexpr span<const size> offsets_span() const { return {offsets_.data(), offsets_.size()}; }
	constexpr span<size> indices_span() { return {indices_.data(), indices_.size()}; }
	constexpr span<const size> indices_span() const { return {indices_.data(), indices_.size()}; }
	constexpr span<scalar> values_span() { return {values_.data(), values_.size()}; }
	constexpr span<const scalar> values_span() const { return {values_.data(), values_.size()}; }

	constexpr size major_size() const { return offsets_.size() - 1; }

	constexpr auto & offsets() { return offsets_; }
	constexpr const auto & offsets() const { return offsets_; }
	constexpr auto & indices() { return indices_; }
	constexpr const auto & indices() const { return indices_; }
	constexpr auto & values() { return values_; }
	constexpr const auto & values() const { return values_; }

protected:
	std::vector<size>   offsets_;
	std::vector<size>   indices_;
	std::vector<scalar> values_;
};

template<class Scalar, class Size>
struct compressed_view_data {
	using scalar = Scalar;
	using size = Size;
	template<class T>
	using span = flecsi::util::span<T>;
	static constexpr bool is_resizable = false;

	compressed_view_data(size msize, size nnz,
	                     const size *offsets, const size * indices,
	                     const scalar *values) :
		major_size_{msize}, nnz_{nnz},
		offsets_{offsets}, indices_{indices}, values_{values} {}

	constexpr const size * offsets_ptr() const { return offsets_; }
	constexpr const size * indices_ptr() const { return indices_; }
	constexpr const scalar * values_ptr() const { return values_; }

	constexpr span<const size> offsets_span() const { return {offsets_, major_size_ + 1 }; }
	constexpr span<const size> indices_span() const { return {indices_, nnz_}; }
	constexpr span<const scalar> values_span() const { return {values_, nnz_}; }

	constexpr size major_size() const { return major_size_; }

protected:
	size major_size_;
	size nnz_;
	const size * offsets_;
	const size * indices_;
	const scalar * values_;
};

enum class major { col, row };

template<class Data, major format>
struct compressed_ops {
	using scalar = typename Data::scalar;
	void spmv(scalar *, const Data &, const scalar *) const {
		flog(error) << "Not implemented" << std::endl;
	}
};

template<class Data>
struct compressed_ops<Data, major::row> {
	using scalar = typename Data::scalar;
	using size = typename Data::size;

	template<class Y, class X>
	void spmv(vec::seq<Y> & y, const Data & data, const vec::seq<X> & x) const {
		const size * rowptr = data.offsets_ptr();
		const size * colind = data.indices_ptr();
		const scalar * values = data.values_ptr();
		for (size i = 0; i < data.major_size(); ++i) {
			y[i] = 0.;
			for (size off = rowptr[i]; off < rowptr[i+1]; ++off) {
				y[i] += values[off] * x[colind[off]];
			}
		}
	}
};

template<class scalar,
         major format = major::row,
         class size = std::size_t,
         class Data = compressed_vector_data<scalar, size>,
         class Ops = compressed_ops<compressed_vector_data<scalar, size>, format>>
struct compressed :
		sparse<compressed<scalar, format, size, Data, Ops>> {
	static constexpr bool is_row_major = format == major::row;

	using base = sparse<compressed<scalar, format, size, Data, Ops>>;
	using base::data;

	template<class T> using span = flecsi::util::span<T>;

	compressed() : major_size_{0}, minor_size_{0}, nnz_{0} {}
	compressed(size rows, size cols, Data d = Data()) :
		base{std::move(d)},
		major_size_{is_row_major ? rows : cols},
		minor_size_{is_row_major ? cols : rows},
		nnz_{data.indices_span().size()} {
		if constexpr (Data::is_resizable) data.resize_major(major_size_);
	}

	constexpr size rows() const { return is_row_major ? major_size_ : minor_size_; }
	constexpr size cols() const { return is_row_major ? minor_size_ : major_size_; }
	constexpr size nnz() const { return nnz_; }

	constexpr size * offsets_ptr() { return data.offsets_ptr(); }
	constexpr const size * offsets_ptr() const { return data.offsets_ptr(); }

	constexpr size * indices_ptr() { return data.indices_ptr(); }
	constexpr const size * indices_ptr() const { return data.indices_ptr(); }

	constexpr scalar * values_ptr() { return data.values_ptr(); }
	constexpr const scalar * values_ptr() const { return data.values_ptr(); }

	constexpr span<size> offsets() { return data.offsets_span(); }
	constexpr span<const size> offsets() const { return data.offsets_span(); }
	constexpr span<size> indices() { return data.indices_span(); }
	constexpr span<const size> indices() const { return data.indices_span(); }
	constexpr span<scalar> values() { return data.values_span(); }
	constexpr span<const scalar> values() const { return data.values_span(); }

	void resize(size nnz) {
		static_assert(Data::is_resizable);
		data.resize_nnz(nnz);
		nnz_ = nnz;
	}

	constexpr void set_nnz(size nnz) { nnz_ = nnz; }

protected:
	size major_size_;
	size minor_size_;
	size nnz_;
};

template<class scalar, major format, class size, class data, class ops>
struct traits<compressed<scalar, format, size, data, ops>> {
	using ops_t = ops;
	using data_t = data;
	using scalar_t = scalar;
	using size_t = size;
};

template<class scalar,
         class size = std::size_t,
         class Data = compressed_vector_data<scalar, size>,
         class Ops =
             compressed_ops<compressed_vector_data<scalar, size>, major::row>>
using csr = compressed<scalar, major::row, size, Data, Ops>;

template<class R, class C, class V>
struct csr_view : compressed<typename V::value_type,
                             major::row,
                             typename R::value_type,
                             compressed_view_data<typename V::value_type, typename R::value_type>,
                             compressed_ops<compressed_view_data<typename V::value_type, typename R::value_type>,major::row>> {
	using base = compressed<
		typename V::value_type,
		major::row,
		typename R::value_type,
		compressed_view_data<typename V::value_type, typename R::value_type>,
		compressed_ops<compressed_view_data<typename V::value_type,
	                                        typename R::value_type>,
	                   major::row>>;
	using data_t = typename base::data_t;

	constexpr csr_view(R r, C c, V v) : base{r.size() - 1, r.size() - 1,
		data_t{r.size() - 1, v.size(), r.data(), c.data(), v.data()}} {}

};


template<class scalar, class size = std::size_t>
struct coo : sparse<coo<scalar, size>> {
	using base = sparse<coo>;
	using base::data;

	coo(size n, size m) :
		rows_{n}, cols_{m} {}

	void resize(size nnz) {
		[=](auto & ... v) { ((v.resize(nnz)), ...); }(data.I, data.J, data.V);
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
		auto rowptr = ret.offsets();
		auto colind = ret.indices();
		auto values = ret.values();
		rowptr[rows_] = 0;
		for (size i = 0; i < nnz(); ++i) {
			++rowptr[I[ind[i]] + 1];
			colind[i] = J[ind[i]];
			values[i] = V[ind[i]];
		}

		for (size i = 0; i < rows(); ++i) {
			rowptr[i+1] += rowptr[i];
		}

		return ret;
	}

protected:
	size rows_;
	size cols_;
};

template<class scalar, class size>
struct traits<coo<scalar, size>> {
	using type = coo<scalar, size>;
	struct data_t {
		std::vector<size> I, J;
		std::vector<scalar> V;
	};
	struct ops_t {
		template<class Y, class X>
		void spmv(vec::seq<Y> &, const data_t &, const vec::seq<X> &) {
			flog(error) << "Not implemented: COO SPMV" << std::endl;
		}
	};
	using scalar_t = scalar;
	using size_t = size;
};
}

namespace flecsi::util::serial {

template<class scalar, class size>
struct traits<flecsolve::mat::compressed_vector_data<scalar, size>> {
	using type = flecsolve::mat::compressed_vector_data<scalar, size>;

	template<class P>
	static void put(P & p, const type & c) {
		serial::put(p,
		            c.offsets(), c.indices(), c.values());
	}

	static type get(const std::byte *& p) {
		type ret;

		ret.offsets() = serial::get<std::vector<size>>(p);
		ret.indices() = serial::get<std::vector<size>>(p);
		ret.values() = serial::get<std::vector<scalar>>(p);

		return ret;
	}
};

template<class scalar,
         flecsolve::mat::major format,
         class size, class data, class ops>
struct traits<flecsolve::mat::compressed<scalar, format, size,
                                         data, ops>> {
	using type = flecsolve::mat::compressed<
		scalar,
		format,
		size,
		data, ops>;

	template<class P>
	static void put(P & p, const type & c) {
		serial::put(p,
		            c.rows(), c.cols(), c.data);
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
