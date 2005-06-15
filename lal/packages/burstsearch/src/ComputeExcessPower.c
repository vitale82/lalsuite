/******** <lalVerbatim file="ComputeExcessPowerCV"> ********
Author: Flanagan, E
$Id$
********* </lalVerbatim> ********/

#include <math.h>

#include <lal/LALRCSID.h>

NRCSID (COMPUTEEXCESSPOWERC, "$Id$");

#include <lal/ExcessPower.h>
#include <lal/Thresholds.h>
#include <lal/XLALError.h>

#define TRUE 1


/******** <lalVerbatim file="ComputeExcessPowerCP"> ********/
int
XLALComputeExcessPower(
	TFTiling *tfTiling,
	const ComputeExcessPowerIn *input,
	const REAL4 *norm
)
/******** </lalVerbatim> ********/
{
	static const char *func = "XLALComputeExcessPower";
	COMPLEX8TimeFrequencyPlane *tfPlane;
	TFTile *tile;
	REAL8 sum;
	REAL8 dof;
	REAL8 numsigma;
	INT4 j;
	INT4 nf;
	INT4 nt;
	INT4 t1;
	INT4 t2;
	INT4 k1;
	INT4 k2;

	/* check on some parameter values */
	if((input->numSigmaMin < 1.0) || (input->alphaDefault < 0.0) || (input->alphaDefault > 1.0) || (tfTiling->numPlanes <= 0))
		XLAL_ERROR(func, XLAL_EDOM);

	/* make sure TF planes have already been computed */
	if(!tfTiling->planesComputed)
		XLAL_ERROR(func, XLAL_EDATA);

	for(tile = tfTiling->firstTile; tile; tile = tile->nextTile) {
		tfPlane = tfTiling->tfp;

		nf = tfPlane->params->freqBins;
		nt = tfPlane->params->timeBins;
		t1 = tile->tstart;
		t2 = tile->tend;
		k1 = tile->fstart;
		k2 = tile->fend;

		if((nf <= 0) || (nt <= 0) ||
		   (t1 < 0) || (t1 > t2) || (t2 > nt) ||
		   (k1 < 0) || (k1 > k2) || (k2 > nf))
			XLAL_ERROR(func, XLAL_EDATA);

		/* Calculate the degrees of freedom of the TF tile */
		dof = 2.0 * (t2 - t1) * tile->deltaT * (k2 - k1) * tile->deltaF;

		sum = 0.0;
		for(j = t1; j < t2; j += (t2 - t1) / dof) {
			INT4 offset = j * nf;
			REAL8 sumnorm = 0.0;
			COMPLEX8 sumz = { 0.0, 0.0 };
			INT4 ii;
			for(ii = k1; ii < k2; ii++) {
				COMPLEX8 z = tfPlane->data[offset+ii];
				sumz.re += z.re;
				sumz.im += z.im;
				sumnorm += norm[ii] * norm[ii];
			}
			sum += (sumz.re*sumz.re + sumz.im*sumz.im)/sumnorm;
		}

		{
		REAL8 rho2;
		rho2 = sum - dof;
		tile->excessPower = rho2;
		numsigma = rho2 / sqrt(2.0 * dof);
		}
		tile->weight = 1.0;

		/* Need to compute an accurate value of likelihood only if
		 * excess power is greater than a few sigma */

		if(numsigma > input->numSigmaMin) {
			tile->firstCutFlag = TRUE;
			tile->alpha = XLALOneMinusChisqCdf(sum, dof);
			if(XLALIsREAL8FailNaN(tile->alpha))
				XLAL_ERROR(func, XLAL_EFUNC);
		} else {
			tile->alpha =  input->alphaDefault; /* default value */
		}
	}

	/* set flag saying alpha for each tile has been computed */
	tfTiling->excessPowerComputed = TRUE;

	/* success */
	return(0);
}
