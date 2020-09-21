#ifdef __cplusplus
#include <vector>
#include <memory>
#include "params.h"
extern "C" {
	double picFloatGpu(double *positionX, double *positionY, double *velocityX, double *velocityY, double *xResult, double *yResult, std::shared_ptr<Params> p);
}
#endif
