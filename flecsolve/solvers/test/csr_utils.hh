#pragma once

#include <fstream>
#include <iostream>

#include "flecsolve/vectors/variable.hh"
#include "flecsolve/vectors/base.hh"

#include "test_mesh.hh"

namespace flecsolve {

using realf = flecsi::field<double>;

using flecsi::na;
using flecsi::ro;
using flecsi::wo;

inline void
init_mesh(std::size_t nrows, testmesh::slot & msh, testmesh::cslot & coloring) {
	std::vector<std::size_t> extents{nrows};
	coloring.allocate(flecsi::processes(), extents);
	msh.allocate(coloring.get());
}

template<class scalar = double, class size = std::size_t>
struct csr {
	csr(size nrows, size nnz)
		: valbuf(std::make_unique<scalar[]>(nnz)),
		  colindbuf(std::make_unique<size[]>(nnz)),
		  rowptrbuf(std::make_unique<size[]>(nrows + 1)), nrows(nrows),
		  nnz(nnz), colind(colindbuf.get()), rowptr(rowptrbuf.get()),
		  values(valbuf.get()) {
		rowptr[0] = 0;
		rowptr[nrows] = nnz;
	}
	csr(size nrows, size nnz, size * colind, size * rowptr, scalar * data)
		: nrows(nrows), nnz(nnz), colind(colind), rowptr(rowptr), valbuf(data) {
		rowptr[0] = 0;
		rowptr[nrows] = nnz;
	}

	void dump(const char * fname) {
		std::ofstream ofile(fname);
		for (size i = 0; i < nrows; i++) {
			for (size off = rowptr[i]; off < rowptr[i + 1]; off++) {
				ofile << "(" << colind[off] << " => " << values[off] << ')';
			}
			ofile << '\n';
		}
	}

protected:
	std::unique_ptr<scalar[]> valbuf;
	std::unique_ptr<size[]> colindbuf;
	std::unique_ptr<size[]> rowptrbuf;

public:
	size nrows, nnz;
	size * colind;
	size * rowptr;
	scalar * values;
};

struct mm_header {
	std::size_t nrows;
	std::size_t ncols;
	std::size_t nnz;
	bool symmetric;
};

inline mm_header read_header(std::ifstream & mfile) {
	mm_header hdr;
	std::array<std::size_t, 3> sizes;
	std::string line;

	hdr.symmetric = false;
	while (std::getline(mfile, line)) {
		if (line.find("symmetric") != std::string::npos) {
			hdr.symmetric = true;
		}
		if (line[0] != '%') {
			std::istringstream iss(line);
			for (int i = 0; i < 3; i++) {
				std::string tok;
				std::getline(iss, tok, ' ');
				sizes[i] = std::atoi(tok.c_str());
			}
			break;
		}
	}

	hdr.nrows = sizes[0];
	hdr.ncols = sizes[1];
	hdr.nnz = sizes[2];

	return hdr;
}

template<std::size_t I>
auto stov(std::string & s) {
	if constexpr (I == 2)
		return std::stof(s);
	else
		return std::stoi(s) - 1;
}
template<class T, std::size_t... I>
void parse_entry_impl(T & t,
                      std::istringstream & iss,
                      std::index_sequence<I...>) {
	std::string tok;
	((std::getline(iss, tok, ' '), std::get<I>(t) = stov<I>(tok)), ...);
}

inline auto parse_entry(std::string & line) {
	std::tuple<std::size_t, std::size_t, double> ret;
	std::istringstream iss(line);

	parse_entry_impl(ret, iss, std::make_index_sequence<3>());

	return ret;
}

inline auto read_mm(const char * fname) {
	std::ifstream mfile(fname);

	auto header = read_header(mfile);

	std::vector<std::size_t> I, J;
	std::vector<double> V;
	[=](auto &... v) { ((v.reserve(header.nnz)), ...); }(I, J, V);

	std::string line;
	while (std::getline(mfile, line)) {
		auto [row, col, val] = parse_entry(line);

		I.push_back(row);
		J.push_back(col);
		V.push_back(val);

		if (header.symmetric and (row != col)) {
			I.push_back(col);
			J.push_back(row);
			V.push_back(val);
		}
	}

	// argsort by rows
	std::vector<std::size_t> ind(I.size());
	std::iota(ind.begin(), ind.end(), 0);
	std::stable_sort(
		ind.begin(), ind.end(), [&](std::size_t i1, std::size_t i2) {
			return I[i1] < I[i2];
		});

	csr mat(header.nrows, ind.size());

	std::size_t curr_row = I[ind[0]];
	for (std::size_t i = 0; i < ind.size(); i++) {
		if (I[ind[i]] != curr_row)
			mat.rowptr[++curr_row] = i;
		mat.colind[i] = J[ind[i]];
		mat.values[i] = V[ind[i]];
	}

	return mat;
};

template<class CSR>
void spmv(const CSR & mat,
          testmesh::accessor<ro> m,
          realf::accessor<ro, na> x,
          realf::accessor<wo, na> y) {
	auto dofs = m.dofs<testmesh::cells>();
	for (std::size_t i = 0; i < mat.nrows; i++) {
		auto dof = dofs[i];
		y[dof] = 0.;
		for (std::size_t off = mat.rowptr[i]; off < mat.rowptr[i + 1]; off++) {
			y[dof] += x[dofs[mat.colind[off]]] * mat.values[off];
		}
	}
}

template<class CSR>
struct csr_op {

	template<class D, class R>
	void apply(const vec::base<D> & x, vec::base<R> & y) const {
		flecsi::execute<spmv<CSR>, flecsi::mpi>(
			mat, x.data.topo(), x.data.ref(), y.data.ref());
	}

	template<class D, class R>
	void residual(const vec::base<D> & b,
	              const vec::base<R> & x,
	              vec::base<R> & r) const {
		apply(x, r);
		r.subtract(b, r);
	}

	template<auto tag, class T>
	auto get_parameters(const T &) const {
		return nullptr;
	}

	auto & get_operator() { return *this; }
	const auto & get_operator() const { return *this; }

	template<class T>
	void reset(const T &) const {}

	CSR mat;

	static constexpr auto input_var = variable<anon_var::anonymous>;
	static constexpr auto output_var = variable<anon_var::anonymous>;
};
template<class CSR>
csr_op(CSR) -> csr_op<CSR>;

}
