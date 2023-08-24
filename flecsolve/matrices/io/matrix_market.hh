#ifndef FLECSOLVE_MATRICES_IO_MATRIX_MARKET_HH
#define FLECSOLVE_MATRICES_IO_MATRIX_MARKET_HH

#include <fstream>
#include <sstream>
#include <string>

#include "flecsi/util/crs.hh"

#include "flecsolve/matrices/seq.hh"

namespace flecsolve::mat::io {

template<class scalar = double, class size = std::size_t>
struct matrix_market {

	struct header {
		size nrows;
		size ncols;
		size nnz;
		bool symmetric;
	};

	static header read_header(std::ifstream & fh) {
		header hdr;
		std::array<size, 3> sizes;
		std::string line;

		hdr.symmetric = false;
		while (std::getline(fh, line)) {
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

	static coo<scalar, size> read(const char * fname) {
		std::ifstream fh(fname);
		auto hdr = read_header(fh);

		return read(fh, hdr);
	}

	static coo<scalar, size> read(std::ifstream & fh, const header & hdr) {
		// only estimate if symmetric
		std::size_t estimate =
			hdr.symmetric ? hdr.nnz * 2 - hdr.nrows : hdr.nnz;
		coo<scalar, size> ret{hdr.nrows, hdr.ncols};
		auto & I = ret.data.I;
		auto & J = ret.data.J;
		auto & V = ret.data.V;

		[=](auto &... v) { ((v.reserve(estimate)), ...); }(I, J, V);

		for (size i = 0; i < hdr.nnz; ++i) {
			std::string line;
			std::getline(fh, line);

			auto [row, col, val] = parse_entry(line);

			I.push_back(row);
			J.push_back(col);
			V.push_back(val);

			if (hdr.symmetric && (row != col)) {
				I.push_back(col);
				J.push_back(row);
				V.push_back(val);
			}
		}

		return ret;
	}

	struct definition {
		definition(const char * fname) : fh{fname} { hdr = read_header(fh); }

		size num_rows() const { return hdr.nrows; }

		size num_cols() const { return hdr.ncols; }

		template<class Range>
		auto graph(const Range & rng) {
			if (!mat_read)
				read_mat();

			flecsi::util::crs conn;

			auto rowptr = mat.offsets();
			auto colind = mat.indices();
			for (auto i : rng) {
				std::vector<size> inds;
				for (size off = rowptr[i]; off < rowptr[i + 1]; ++off) {
					inds.push_back(colind[off]);
				}
				conn.add_row(inds);
			}

			return conn;
		}

		template<class Range>
		auto matrix(const Range & rng) {
			if (!mat_read)
				read_mat();
			csr<scalar, size> ret{rng.size(), rng.size()};

			// should really do some more reserving
			auto & ret_data = ret.data;
			auto & rowptr = ret_data.offsets();
			auto & colind = ret_data.indices();
			auto & values = ret_data.values();
			size nnz = 0;
			size ii = 0;
			for (auto i : rng) {
				for (size off = mat.offsets()[i]; off < mat.offsets()[i + 1];
				     ++off) {
					colind.push_back(mat.indices()[off]);
					values.push_back(mat.values()[off]);
					++nnz;
				}

				rowptr[ii++] = mat.offsets()[i] - mat.offsets()[rng.front()];
			}
			rowptr[ii] =
				mat.offsets()[rng.back() + 1] - mat.offsets()[rng.front()];
			ret.set_nnz(nnz);

			return ret;
		}

	protected:
		void read_mat() {
			mat = read(fh, hdr).tocsr();
			mat_read = true;
		}

		csr<scalar, size> mat;
		header hdr;
		std::ifstream fh;
		bool mat_read = false;
	};

protected:
	template<std::size_t I>
	static auto stov(std::string & s) {
		if constexpr (I == 2)
			return std::stof(s);
		else
			return std::stoi(s) - 1;
	}
	template<class T, std::size_t... I>
	static void parse_entry_impl(T & t,
	                             std::istringstream & iss,
	                             std::index_sequence<I...>) {
		std::string tok;
		((std::getline(iss, tok, ' '), std::get<I>(t) = stov<I>(tok)), ...);
	}

	static auto parse_entry(std::string & line) {
		std::tuple<size, size, scalar> ret;
		std::istringstream iss(line);

		parse_entry_impl(ret, iss, std::make_index_sequence<3>());

		return ret;
	}
};
}

#endif
