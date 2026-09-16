/*
© 2025. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract
89233218CNA000001 for Los Alamos National Laboratory (LANL), which is
operated by Triad National Security, LLC for the U.S. Department of
Energy/National Nuclear Security Administration. All rights in the
program are reserved by Triad National Security, LLC, and the
U.S. Department of Energy/National Nuclear Security Administration.
The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material
to reproduce, prepare. derivative works, distribute copies to the
public, perform publicly and display publicly, and to permit others to
do so.
*/
#ifndef FLECSOLVE_TIME_INTEGRATOR_EXPLICIT_RK_H
#define FLECSOLVE_TIME_INTEGRATOR_EXPLICIT_RK_H

#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>

namespace flecsolve::time_integrator::detail {

template<auto const & Tableau, std::size_t Row, std::size_t Index = 0>
consteval std::size_t a_nonzero_count() {
	if constexpr (Index == Tableau.a[Row].size())
		return 0;
	else
		return (Tableau.a[Row][Index] != 0.) +
		       a_nonzero_count<Tableau, Row, Index + 1>();
}

template<auto const & Tableau,
         std::size_t Row,
         std::size_t Number,
         std::size_t Index = 0>
consteval std::size_t a_nth_nonzero() {
	if constexpr (Tableau.a[Row][Index] != 0.) {
		if constexpr (Number == 0)
			return Index;
		else
			return a_nth_nonzero<Tableau, Row, Number - 1, Index + 1>();
	}
	else {
		return a_nth_nonzero<Tableau, Row, Number, Index + 1>();
	}
}

template<auto const & Tableau,
         bool Embedded,
         std::size_t Index = 0>
consteval std::size_t b_nonzero_count() {
	if constexpr (Index == Tableau.b.size())
		return 0;
	else
		return ((Embedded ? Tableau.b_embedded[Index] : Tableau.b[Index]) != 0.) +
		       b_nonzero_count<Tableau, Embedded, Index + 1>();
}

template<auto const & Tableau,
         bool Embedded,
         std::size_t Number,
         std::size_t Index = 0>
consteval std::size_t b_nth_nonzero() {
	if constexpr ((Embedded ? Tableau.b_embedded[Index] : Tableau.b[Index]) !=
	              0.) {
		if constexpr (Number == 0)
			return Index;
		else
			return b_nth_nonzero<Tableau, Embedded, Number - 1, Index + 1>();
	}
	else {
		return b_nth_nonzero<Tableau, Embedded, Number, Index + 1>();
	}
}

template<auto const & Tableau,
         std::size_t Stage,
         std::size_t Previous,
         class DeltaT,
         class Work>
void append_a_terms(DeltaT & dt, Work & work) {
	if constexpr (Previous < Stage) {
		if constexpr (Tableau.a[Stage][Previous] != 0.) {
			std::get<Tableau.a.size() + 1>(work).axpy(
				Tableau.a[Stage][Previous] * dt,
				std::get<Previous>(work),
				std::get<Tableau.a.size() + 1>(work));
		}
		append_a_terms<Tableau, Stage, Previous + 1>(dt, work);
	}
}

template<auto const & Tableau,
         bool Embedded,
         std::size_t Previous,
         class DeltaT,
         class Work>
void append_b_terms(DeltaT & dt, Work & work) {
	if constexpr (Previous < Tableau.b.size()) {
		constexpr auto coefficient =
			Embedded ? Tableau.b_embedded[Previous] : Tableau.b[Previous];
		if constexpr (coefficient != 0.) {
			std::get<Tableau.b.size() + 1>(work).axpy(
				coefficient * dt,
				std::get<Previous>(work),
				std::get<Tableau.b.size() + 1>(work));
		}
		append_b_terms<Tableau, Embedded, Previous + 1>(dt, work);
	}
}

template<auto const & Tableau,
         std::size_t Previous,
         class DeltaT,
         class Work>
void append_error_terms(DeltaT & dt, Work & work) {
	if constexpr (Previous < Tableau.b.size()) {
		if constexpr (Tableau.b[Previous] - Tableau.b_embedded[Previous] != 0.)
			std::get<Tableau.b.size()>(work).axpy(
				(Tableau.b[Previous] - Tableau.b_embedded[Previous]) * dt,
				std::get<Previous>(work),
				std::get<Tableau.b.size()>(work));
		append_error_terms<Tableau, Previous + 1>(dt, work);
	}
}

template<auto const & Tableau,
         std::size_t Stage,
         class DeltaT,
         class Curr,
         class Work>
void make_stage(DeltaT & dt, Curr & curr, Work & work) {
	constexpr auto count = a_nonzero_count<Tableau, Stage>();
	auto & next = std::get<Tableau.a.size() + 1>(work);

	if constexpr (count == 0) {
		next.copy(curr);
	}
	else if constexpr (count == 1) {
		constexpr auto first = a_nth_nonzero<Tableau, Stage, 0>();
		next.axpy(Tableau.a[Stage][first] * dt,
		          std::get<first>(work),
		          curr);
	}
	else {
		constexpr auto first = a_nth_nonzero<Tableau, Stage, 0>();
		constexpr auto second = a_nth_nonzero<Tableau, Stage, 1>();
		next.linear_sum(Tableau.a[Stage][first] * dt,
		                std::get<first>(work),
		                Tableau.a[Stage][second] * dt,
		                std::get<second>(work));
		append_a_terms<Tableau, Stage, second + 1>(dt, work);
		next.axpy(1., next, curr);
	}
}

template<auto const & Tableau, bool Embedded, class DeltaT, class Curr, class Work>
void make_solution(DeltaT & dt, Curr & curr, Work & work) {
	constexpr auto count = b_nonzero_count<Tableau, Embedded>();
	auto & next = std::get<Tableau.b.size() + 1>(work);
	static_assert(count > 0, "Runge-Kutta solution weights cannot be empty");

	if constexpr (count == 1) {
		constexpr auto first = b_nth_nonzero<Tableau, Embedded, 0>();
		next.axpy((Embedded ? Tableau.b_embedded[first] : Tableau.b[first]) * dt,
		          std::get<first>(work),
		          curr);
	}
	else {
		constexpr auto first = b_nth_nonzero<Tableau, Embedded, 0>();
		constexpr auto second = b_nth_nonzero<Tableau, Embedded, 1>();
		next.linear_sum(
			(Embedded ? Tableau.b_embedded[first] : Tableau.b[first]) * dt,
			std::get<first>(work),
			(Embedded ? Tableau.b_embedded[second] : Tableau.b[second]) * dt,
			std::get<second>(work));
		append_b_terms<Tableau, Embedded, second + 1>(dt, work);
		next.axpy(1., next, curr);
	}
}

template<auto const & Tableau, class DeltaT, class Work>
void make_error(DeltaT & dt, Work & work) {
	constexpr auto count = [] {
		std::size_t result = 0;
		for (std::size_t i = 0; i < Tableau.b.size(); ++i)
			result += (Tableau.b[i] - Tableau.b_embedded[i]) != 0.;
		return result;
	}();
	static_assert(count > 0, "Runge-Kutta error weights cannot be empty");
	auto & error = std::get<Tableau.b.size()>(work);

	constexpr auto first = [] {
		for (std::size_t i = 0; i < Tableau.b.size(); ++i)
			if (Tableau.b[i] - Tableau.b_embedded[i] != 0.)
				return i;
		return Tableau.b.size();
	}();
	if constexpr (count == 1) {
		error.scale((Tableau.b[first] - Tableau.b_embedded[first]) * dt,
		            std::get<first>(work));
	}
	else {
		constexpr auto second = [] {
			std::size_t found = 0;
			for (std::size_t i = first + 1; i < Tableau.b.size(); ++i)
				if (Tableau.b[i] - Tableau.b_embedded[i] != 0. && ++found == 1)
					return i;
			return Tableau.b.size();
		}();
		error.linear_sum((Tableau.b[first] - Tableau.b_embedded[first]) * dt,
		                 std::get<first>(work),
		                 (Tableau.b[second] - Tableau.b_embedded[second]) * dt,
		                 std::get<second>(work));
		append_error_terms<Tableau, second + 1>(dt, work);
	}
}

template<auto const & Tableau,
         std::size_t Stage = 0,
         class Op,
         class DeltaT,
         class Curr,
         class Out,
         class Work>
void advance_stages(Op & op, DeltaT & dt, Curr & curr, Out & out, Work & work) {
	if constexpr (Stage < Tableau.a.size()) {
		if constexpr (Stage == 0)
			op.apply(curr, std::get<Stage>(work));
		else {
			make_stage<Tableau, Stage>(dt, curr, work);
			op.apply(std::get<Tableau.a.size() + 1>(work),
			          std::get<Stage>(work));
		}
		advance_stages<Tableau, Stage + 1>(op, dt, curr, out, work);
	}
	else {
		constexpr bool final_stage_is_solution = [] {
			for (std::size_t i = 0; i < Tableau.b.size(); ++i)
				if (Tableau.b[i] != Tableau.a[Tableau.a.size() - 1][i])
					return false;
			return true;
		}();
		if constexpr (final_stage_is_solution)
			out.copy(std::get<Tableau.a.size() - 1>(work));
		else {
			make_solution<Tableau, false>(dt, curr, work);
			out.copy(std::get<Tableau.a.size() + 1>(work));
		}
		make_error<Tableau>(dt, work);
	}
}

template<auto const & Tableau, class Op, class DeltaT, class Curr, class Out, class Work>
void advance(Op & op, DeltaT dt, Curr & curr, Out & out, Work & work) {
	advance_stages<Tableau>(op, dt, curr, out, work);
}

}

#endif
