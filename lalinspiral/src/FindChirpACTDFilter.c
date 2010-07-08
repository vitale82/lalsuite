#if 0
<lalVerbatim file="FindChirpACTDFilterCV">
Author: Brown D. A., McKechan D. J. A.
$Id$
</lalVerbatim>

<lalLaTeX>
\subsection{Module \texttt{FindChirpACTDFilter.c}}
\label{ss:FindChirpACTDFilter.c}

This module provides the core of the matched filter for amplitude corrected
binary inspiral chirps.


\subsubsection*{Prototypes}
\vspace{0.1in}
\input{FindChirpACTDFilterCP}
\idx{LALFindChirpACTDFilterSegment()}

\subsubsection*{Description}

\subsubsection*{Algorithm}

\subsubsection*{Notes}

\vfill{\footnotesize\input{FindChirpACTDFilterCV}}
</lalLaTeX>
#endif

#include <math.h>
#include <lal/LALStdio.h>
#include <lal/LALStdlib.h>
#include <lal/LALError.h>
#include <lal/LALConstants.h>
#include <lal/Date.h>
#include <lal/AVFactories.h>
#include <lal/FindChirp.h>
#include <lal/FindChirpChisq.h>
#include <lal/FindChirpACTD.h>

double rint(double x);
/* debugging */
extern int vrbflg;                      /* verbocity of lal function    */


NRCSID (FINDCHIRPACTDFILTERC, "$Id$");


/* <lalVerbatim file="FindChirpACTDFilterCP"> */
void
LALFindChirpACTDFilterSegment (
    LALStatus                  *status,
    SnglInspiralTable         **eventList,
    FindChirpFilterInput       *input,
    FindChirpFilterParams      *params
    )
/* </lalVerbatim> */
{
  UINT4                 i, j, k, kmax;
  UINT4                 numPoints;
  UINT4                 tmpltLength;
  UINT4                 deltaEventIndex;
  UINT4                 ignoreIndex;
  REAL8                 deltaT;
  REAL8                 deltaF;
  REAL4                 normFac;
  REAL4                 normFacSq;
  COMPLEX8Vector      **qtilde;
  REAL4Vector         **q;
  COMPLEX8             *inputData     = NULL;
  COMPLEX8Vector        tmpltSignal[NACTDTILDEVECS];

  /* For use in applying the constraint */
  REAL4Vector          *qForConstraint = NULL;
  UINT4                matrixDim;
  /*
  SnglInspiralTable    *thisEvent     = NULL;
  REAL4                 modqsqThresh;
  BOOLEAN               haveEvent     = 0;
  */

  INITSTATUS( status, "LALFindChirpACTDFilter", FINDCHIRPACTDFILTERC );
  ATTATCHSTATUSPTR( status );


  /*
   *
   * check that the arguments are reasonable
   *
   */


  /* make sure the output handle exists, but points to a null pointer */
  ASSERT( eventList, status, FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL );
  ASSERT( !*eventList, status, FINDCHIRPH_ENNUL, FINDCHIRPH_MSGENNUL );

  /* make sure that the parameter structure exists */
  ASSERT( params, status, FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL );

  /* check that the filter parameters are reasonable */
  ASSERT( params->deltaT > 0, status,
      FINDCHIRPH_EDTZO, FINDCHIRPH_MSGEDTZO );
  ASSERT( params->rhosqThresh >= 0, status,
      FINDCHIRPH_ERHOT, FINDCHIRPH_MSGERHOT );
  ASSERT( params->chisqThresh >= 0, status,
      FINDCHIRPH_ECHIT, FINDCHIRPH_MSGECHIT );
  ASSERT( params->chisqDelta >= 0, status,
      FINDCHIRPH_ECHIT, FINDCHIRPH_MSGECHIT );

  /* check that the fft plan exists */
  ASSERT( params->invPlanACTD, status, FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL );

  /* check that the workspace vectors exist */
  ASSERT( params->qVecACTD, status, FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL );
  ASSERT( params->qtildeVecACTD, status, FINDCHIRPH_ENULL,
          FINDCHIRPH_MSGENULL );

  /* check that the chisq parameter and input structures exist */
  ASSERT( params->chisqParams, status, FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL );
  ASSERT( params->chisqInput, status, FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL );

  /* if a rhosqVec vector has been created, check we can store data in it */
  if ( params->rhosqVec )
  {
    ASSERT( params->rhosqVec->data->data, status,
        FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL );
    ASSERT( params->rhosqVec->data, status,
        FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL );
  }

  if ( params->cVec )
  {
    ASSERT( params->cVec->data->data, status,
        FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL  );
    ASSERT( params->cVec->data, status,
        FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL );
  }

  /* if we are doing a chisq, check we can store the data */
  if ( input->segment->chisqBinVec->length )
  {
    ASSERT( params->chisqVec, status,
        FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL );
    ASSERT( params->chisqVec->data, status,
        FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL );
  }

  /* make sure that the input structure exists */
  ASSERT( input, status, FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL );

  /* make sure that the input structure contains some input */
  ASSERT( input->fcTmplt, status, FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL );
  ASSERT( input->segment, status, FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL );

  /* check the approximant */
  if ( params->approximant != AmpCorPPN )
  {
    ABORT( status, FINDCHIRPH_EUAPX, FINDCHIRPH_MSGEUAPX );
  }

  /* make sure the approximant in the tmplt and segment agree */
  if ( params->approximant != input->fcTmplt->tmplt.approximant ||
      params->approximant != input->segment->approximant )
  {
    ABORT( status, FINDCHIRPH_EAPRX, FINDCHIRPH_MSGEAPRX );
  }



  /*
   *
   * point local pointers to input and output pointers
   *
   */

  /* number of points in a segment */
  numPoints = params->qVecACTD[0]->length;
  tmpltLength = input->fcTmplt->ACTDtilde->vectorLength;

  /* Create the vector for the constraint */
  qForConstraint = XLALCreateREAL4Vector( NACTDTILDEVECS );

  /* workspace vectors */
  qtilde = params->qtildeVecACTD;
  q      = params->qVecACTD;

  for( i = 0; i < NACTDTILDEVECS; ++i )
  {
    /* filter */
	  memset( qtilde[i]->data, 0, qtilde[i]->length * sizeof( COMPLEX8 ) );
    memset( q[i]->data, 0, numPoints * sizeof( REAL4 ) );

		/* template */
    tmpltSignal[i].length = tmpltLength;
    tmpltSignal[i].data   = input->fcTmplt->ACTDtilde->data + i * tmpltLength;
  }


  /* data */
  inputData = input->segment->data->data->data;


  deltaT = params->deltaT;

  deltaF = 1.0 / ( deltaT * (REAL8)(numPoints)  );
  /*kmax = input->fcTmplt->tmplt.fFinal / deltaF < numPoints/2 ?
  input->fcTmplt->tmplt.fFinal / deltaF : numPoints/2;*/
  /* Craig: I suspect the limits of integration are incorrect. I will
     artificially set it to numPoints/2. */
  kmax = numPoints / 2;

  /* Calculate the dimension of the matrix used in the constraint */
  matrixDim = 2 * (input->fcTmplt->stopVecACTD - input->fcTmplt->startVecACTD );

  /*
   *
   * compute viable search regions in the snrsq vector
   *
   */


  if ( input->fcTmplt->tmplt.tC <= 0 )
  {
    ABORT( status, FINDCHIRPH_ECHTZ, FINDCHIRPH_MSGECHTZ );
  }

  deltaEventIndex = (UINT4) rint( (input->fcTmplt->tmplt.tC / deltaT) + 1.0 );

  /* ignore corrupted data at start and end */
  params->ignoreIndex = ignoreIndex =
    ( input->segment->invSpecTrunc / 2 ) + deltaEventIndex;

  if ( lalDebugLevel & LALINFO )
  {
    CHAR infomsg[256];

    snprintf( infomsg, sizeof(infomsg) / sizeof(*infomsg),
        "m1 = %e m2 = %e => %e seconds => %d points\n"
        "invSpecTrunc = %d => ignoreIndex = %d\n",
        input->fcTmplt->tmplt.mass1, input->fcTmplt->tmplt.mass2,
        input->fcTmplt->tmplt.tC , deltaEventIndex,
        input->segment->invSpecTrunc, ignoreIndex );
    LALInfo( status, infomsg );
  }

  /* XXX check that we are not filtering corrupted data XXX */
  /* XXX this is hardwired to 1/4 segment length        XXX */
  if ( ignoreIndex > numPoints / 4 )
  {
    ABORT( status, FINDCHIRPH_ECRUP, FINDCHIRPH_MSGECRUP );
  }
  /* XXX reset ignoreIndex to one quarter of a segment XXX */
  params->ignoreIndex = ignoreIndex = numPoints / 4;

  if ( lalDebugLevel & LALINFO )
  {
    CHAR infomsg[256];

    snprintf( infomsg, sizeof(infomsg) / sizeof(*infomsg),
        "filtering from %d to %d\n",
        ignoreIndex, numPoints - ignoreIndex );
    LALInfo( status, infomsg );
  }


  /*
   *
   * compute qtildes and qs
   *
   */


  /* qtilde positive frequency, not DC or nyquist */
  for( i = 0; i < NACTDTILDEVECS; ++i )
  {
    if ( (i % NACTDVECS >= input->fcTmplt->startVecACTD )
      && (i % NACTDVECS < input->fcTmplt->stopVecACTD ) )
    {
      for ( k = 1; k < kmax; ++k )
      {
        REAL4 r = inputData[k].re;
        REAL4 s = inputData[k].im;
        REAL4 x = tmpltSignal[i].data[k].re;
        REAL4 y = 0 - tmpltSignal[i].data[k].im; /* NB: Complex conj. */

        qtilde[i]->data[k].re = r*x - s*y;
        qtilde[i]->data[k].im = r*y + s*x;
      }

      /* inverse fft to get q */
      if ( XLALREAL4ReverseFFT( q[i], qtilde[i], params->invPlanACTD ) == 
                                                                XLAL_FAILURE )
      {
        ABORTXLAL( status );
      }
    }
  }




  /*
   *
   * calculate signal to noise squared
   *
   */


  /* if full snrsq vector is required, set it to zero */
  if ( params->rhosqVec )
    memset( params->rhosqVec->data->data, 0, numPoints * sizeof( REAL4 ) );

  if (params->cVec )
    memset( params->cVec->data->data, 0, numPoints * sizeof( COMPLEX8 ) );


  /* normalisation */
  normFac = 2.0 * deltaT / (REAL4)(numPoints);
	normFacSq = normFac * normFac;

	/* normalised snr threhold */
  /*
	modqsqThresh = params->rhosqThresh / norm;
  */

  /* if full snrsq vector is required, store the snrsq */
  if ( params->rhosqVec )
  {
    memcpy( params->rhosqVec->name, input->segment->data->name,
        LALNameLength * sizeof(CHAR) );
    memcpy( &(params->rhosqVec->epoch), &(input->segment->data->epoch),
        sizeof(LIGOTimeGPS) );
    params->rhosqVec->deltaT = input->segment->deltaT;

    for ( j = 0; j < numPoints; ++j )
    {
      REAL4 rhoSq = 0.0;
      
      for( i = 0; i < NACTDTILDEVECS; ++i )
      {
        rhoSq += q[i]->data[j] * q[i]->data[j];
      }

      params->rhosqVec->data->data[j] = rhoSq * normFacSq ;

      if( input->fcTmplt->tmplt.ACTDconstraint == 1 )
      {
        /* Go through the transformation matrix */
        memset( qForConstraint->data, 0, NACTDTILDEVECS * sizeof( REAL4 ));
        for ( i = 0; i < NACTDTILDEVECS; i++ )
        {
          for ( k = 0; k < NACTDTILDEVECS; k++ )
          {
            if ( ( i % NACTDVECS - input->fcTmplt->startVecACTD 
                  < matrixDim / 2 )
              && ( k % NACTDVECS - input->fcTmplt->startVecACTD 
                  < matrixDim / 2 ) )
            {
              UINT4 idx1, idx2, nVec;

              nVec = matrixDim / 2;
              idx1 = (i / NACTDVECS) * nVec + i % NACTDVECS 
                      - input->fcTmplt->startVecACTD;
              idx2 = (k / NACTDVECS) * nVec + k % NACTDVECS 
                      - input->fcTmplt->startVecACTD;
              qForConstraint->data[i] += gsl_matrix_get( 
                input->fcTmplt->ACTDconmatrix, idx1, idx2 ) * q[k]->data[j];
            }
          }
        }
        if ( sqrt( (qForConstraint->data[0] * qForConstraint->data[0]
                  + qForConstraint->data[3] * qForConstraint->data[3])
                 / (qForConstraint->data[1] * qForConstraint->data[1]
                  + qForConstraint->data[4] * qForConstraint->data[4]))
                  > input->fcTmplt->ACTDconstraint->data[0] )
        {
          params->rhosqVec->data->data[j] = 0.0 ;
        }

        if ( sqrt( (qForConstraint->data[2] * qForConstraint->data[2]
                  + qForConstraint->data[5] * qForConstraint->data[5])
                 / (qForConstraint->data[1] * qForConstraint->data[1]
                  + qForConstraint->data[4] * qForConstraint->data[4]))
                  > input->fcTmplt->ACTDconstraint->data[2] )
        {
          params->rhosqVec->data->data[j] = 0.0 ;
        }
      }
    }
  }

/*
  if ( params->cVec )
  {
    memcpy( params->cVec->name, input->segment->data->name,
        LALNameLength * sizeof(CHAR) );
    memcpy( &(params->cVec->epoch), &(input->segment->data->epoch),
        sizeof(LIGOTimeGPS) );
    params->cVec->deltaT = input->segment->deltaT;

    for ( j = 0; j < numPoints; ++j )
    {

      params->cVec->data->data[j].re = sqrt(norm) * q[j].re;
      params->cVec->data->data[j].im = sqrt(norm) * q[j].im;

    }
  }
*/

  #if 0
  /* This is done in FindChirpClusterEvents now!!*/
  /* determine if we need to compute the chisq vector */
  if ( input->segment->chisqBinVec->length )
  {
    /* look for an event in the filter output */
    for ( j = ignoreIndex; j < numPoints - ignoreIndex; ++j )
    {
      REAL4 modqsq = q[j].re * q[j].re + q[j].im * q[j].im;

      /* if snrsq exceeds threshold at any point */
      if ( modqsq > modqsqThresh )
      {
        haveEvent = 1;        /* mark segment to have events    */
        break;
      }
    }
    if ( haveEvent )
    {
      /* compute the chisq vector for this segment */
      memset( params->chisqVec->data, 0,
          params->chisqVec->length * sizeof(REAL4) );

      /* pointers to chisq input */
      params->chisqInput->qtildeVec = params->qtildeVec;
      params->chisqInput->qVec      = params->qVec;

      /* pointer to the chisq bin vector in the segment */
      params->chisqParams->chisqBinVec = input->segment->chisqBinVec;
      params->chisqParams->norm        = norm;

      /* compute the chisq bin boundaries for this template */
      if ( ! params->chisqParams->chisqBinVec->data )
      {
        LALFindChirpComputeChisqBins( status->statusPtr,
            params->chisqParams->chisqBinVec, input->segment, kmax );
        CHECKSTATUSPTR( status );
      }

      /* compute the chisq threshold: this is slow! */
      LALFindChirpChisqVeto( status->statusPtr, params->chisqVec,
          params->chisqInput, params->chisqParams );
      CHECKSTATUSPTR (status);
    }
  }
  #endif

  /* Free memory */
  XLALDestroyREAL4Vector( qForConstraint );

  /* normal exit */
  DETATCHSTATUSPTR( status );
  RETURN( status );
}
