/******** <lalVerbatim file="CreateTFTilingCV"> ********
Author: Eanna Flanagan, and Cannon, K.
$Id$
********* </lalVerbatim> **********/

#include <math.h>
#include <lal/LALRCSID.h>

NRCSID(CREATETFTILINGC, "$Id$");

#include <lal/LALStdlib.h>
#include <lal/TFTransform.h>
#include <lal/XLALError.h>


/*
 * The number of degrees of freedom in a tile.
 */

REAL8 XLALTFTileDegreesOfFreedom(const TFTile *tile)
{
	return((2 * tile->tbins * tile->channels) * tile->deltaT * tile->deltaF);
}


/*
 * Macro for looping over all tiles.  This is ugly but it ensures that the
 * initialization, increment, and terminate statements are the same in the two
 * places the loop is done.
 */

#define FOR_EACH_TILE \
	for(tbins = 2; tbins <= max_tbins; tbins *= 2) \
		for(channels = 1 / (tbins * plane_deltaT * plane_deltaF); channels <= max_channels; channels *= 2) \
			for(tstart = 0; tstart + tbins <= plane_channel_length; tstart += tbins / inv_fractional_stride) \
				for(channel0 = 0; channel0 + channels <= plane_num_channels; channel0 += channels / inv_fractional_stride)



/******** <lalVerbatim file="CreateTFTilingCP"> ********/
TFTiling *XLALCreateTFTiling(
	REAL8 plane_deltaT,
	REAL8 plane_flow,
	REAL8 plane_deltaF,
	UINT4 plane_channel_length,
	UINT4 plane_num_channels,
	INT4 inv_fractional_stride,
	REAL8 maxTileBandwidth,
	REAL8 maxTileDuration
)
/******** </lalVerbatim> *********/
{
	const char func[] = "XLALCreateTFTiling";
	TFTiling *tiling;
	TFTile *tile;
	int *weight;
	int numtiles;

	/* tile size limits */
	unsigned max_tbins;
	unsigned max_channels;
	int maxDOF;

	/* coordinates of a TF tile */
	unsigned channel0;
	unsigned channels;
	unsigned tstart;
	unsigned tbins;

	/* determine the tile size limits */
	max_tbins = maxTileDuration / plane_deltaT;
	max_channels = maxTileBandwidth / plane_deltaF;
	if((plane_channel_length < max_tbins) || (plane_num_channels < max_channels))
		XLAL_ERROR_NULL(func, XLAL_EINVAL);
	maxDOF = (2 * max_tbins * max_channels) * plane_deltaT * plane_deltaF;

	/* Count the tiles */
	numtiles = 0;
	FOR_EACH_TILE {
		numtiles++;
	}
	if(!numtiles)
		/* can't fit any tiles into the TF plane! */
		XLAL_ERROR_NULL(func, XLAL_EINVAL);

	/* allocate memory */
	tiling = LALMalloc(sizeof(*tiling));
	tile = LALMalloc(numtiles * sizeof(*tile));
	weight = LALCalloc(maxDOF + 1, sizeof(*weight));
	if(!tiling || !tile || !weight) {
		LALFree(tiling);
		LALFree(tile);
		LALFree(weight);
		XLAL_ERROR_NULL(func, XLAL_ENOMEM);
	}
	tiling->tile = tile;
	tiling->numtiles = numtiles;

	/* initialize each tile */
	FOR_EACH_TILE {
		tile->channel0 = channel0;
		tile->channels = channels;
		tile->tstart = tstart;
		tile->tbins = tbins;
		tile->flow = plane_flow;
		tile->deltaT = plane_deltaT;
		tile->deltaF = plane_deltaF;
		tile->excessPower = XLAL_REAL8_FAIL_NAN;
		tile->confidence = XLAL_REAL8_FAIL_NAN;
		weight[(int) XLALTFTileDegreesOfFreedom(tile)]++;
		tile++;
	}

	/* determine the weighting for each tile */
	for(tile = tiling->tile; numtiles; numtiles--, tile++)
		tile->lnweight = log(weight[(int) XLALTFTileDegreesOfFreedom(tile)]);
	LALFree(weight);

	return(tiling);
}


/******** <lalVerbatim file="DestroyTFTilingCP"> ********/
void
XLALDestroyTFTiling(
	TFTiling *tiling
)
/******** </lalVerbatim> ********/
{
	if(tiling)
		LALFree(tiling->tile);
	LALFree(tiling);
}
