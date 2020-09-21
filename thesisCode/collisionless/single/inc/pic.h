#ifdef __cplusplus
#include <vector>
#include <memory>
#include "params.h"
extern "C" {
	float picFloatGpu(float *positionX, float *positionY, float *velocityX, float *velocityY, float *xResult, float *yResult, std::shared_ptr<Params> p);
}
#endif
