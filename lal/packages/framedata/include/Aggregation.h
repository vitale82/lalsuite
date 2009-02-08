/*
 * Aggregation.h - online frame data aggregation rountines
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with with program; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 * MA  02111-1307  USA
 *
 * Copyright (C) 2009 Adam Mercer
 */

#ifndef _AGGREGATION_H
#define _AGGREGATION_H

#include <lal/LALDatatypes.h>
#include <lal/FrameCache.h>
#include <lal/FrameStream.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ONLINE_FRAME_DURATION 16
#define ONLINE_STRAIN_CHANNEL "DMT-STRAIN"
#define ONLINE_STATE_VECTOR "DMT-STATE_VECTOR"
#define ONLINE_DQ_VECTOR "DMT-DATA_QUALITY_VECTOR"
#define ONLINE_SAMPLE_RATE 16384


/* function prototypes */
static INT4 return_frame_start(LIGOTimeGPS *gps);
LIGOTimeGPS *XLALAggregationFrameStart(LIGOTimeGPS *gps);
CHAR *XLALAggregationDirectoryPath(CHAR *ifo, LIGOTimeGPS *gps);
CHAR *XLALAggregationFrameType(CHAR *ifo);
CHAR *XLALAggregationFrameFilename(CHAR *ifo, LIGOTimeGPS *gps);
CHAR *XLALAggregationFramePathFilename(CHAR *ifo, LIGOTimeGPS *gps);
CHAR *XLALAggregationFrameURL(CHAR *ifo, LIGOTimeGPS *gps);
FrCache *XLALAggregationFrameCache(CHAR *ifo, LIGOTimeGPS *start, INT4 length);
FrStream *XLALAggregationFrameStream(CHAR *ifo, LIGOTimeGPS *start, INT4 length);
REAL8TimeSeries *XLALAggregationStrainData(CHAR *ifo, LIGOTimeGPS *start, INT4 length);
INT4TimeSeries *XLALAggregationDQVector(CHAR *ifo, LIGOTimeGPS *start, INT4 length);

#ifdef __cplusplus
}
#endif

#endif
