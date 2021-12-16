#pragma once

#include <petscvec.h>

#include <flecsi/data.hh>

namespace flecsi::linalg {


struct petsc_data
{
	using field_t = field<Vec, data::single>;
	using field_definition = field_t::definition<topo::index>;
	using len_t = std::size_t;
	using real_t = PetscScalar;

	template<flecsi::partition_privilege_t priv>
	using acc = field_t::accessor<priv>;

	template<class OtherVec>
	petsc_data(const field_definition & def, MPI_Comm comm, const OtherVec & v) : def(def) {
		using vec_data_t = typename OtherVec::data_t;
		using data_util = typename vec_data_t::util;
		using topo_acc = typename vec_data_t::topo_acc;
		execute<tasks<data_util>::template create_petsc_vec<topo_acc>, mpi>(
			def(process_topology), comm, v.data.topo,
			v.global_size().get());
	}

	const field_definition & def;

	auto ref() const { return def(process_topology); }
	auto fid() const { return ref().fid(); }

protected:
	template<class Util>
	struct tasks
	{
		template<class TopoAcc>
		static PetscErrorCode create_petsc_vec(acc<wo> ac, MPI_Comm comm, TopoAcc m, len_t global_size) {
			PetscErrorCode ierr;

			Vec v;
			auto n = Util::dofs(m).size();
			ierr = VecCreateMPI(comm, n, global_size, &v);CHKERRQ(ierr);
			ac = v;

			return 0;
		}
	};
};

}
