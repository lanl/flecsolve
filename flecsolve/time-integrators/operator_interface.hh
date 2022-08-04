#ifndef FLECSI_LINALG_TIME_INTEGRATOR_OPERATOR_INTERFACE_H
#define FLECSI_LINALG_TIME_INTEGRATOR_OPERATOR_INTERFACE_H

namespace flecsolve::time_integrator {

struct operator_interface {
	operator_interface
	double scaling;
};

}
#endif
