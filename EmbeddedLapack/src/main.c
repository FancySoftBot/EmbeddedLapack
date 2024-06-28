#include <time.h>

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "f2c.h"
#include "clapack.h"

#define ROWS 3
#define COLS 3
#define BUF_SIZE 64

typedef struct LinalgBuffer
{
    int len;
    int max_len;
    doublereal data[BUF_SIZE];
} LinalgBuffer;


/**
 * @brief      Computes SVD of matrix A with buffer (mxn)
 *
 * @details    Ufull and Vfull affect the output matrices as follows:
 *             if (m <= n) then s = m
 *             else s = n
 *             if (Ufull, Vfull) == (0, 0) then U(m,s), S(s,s), VT(s,n)
 *             if (Ufull, Vfull) == (1, 0) then U(m,m), S(m,s), VT(s,n)
 *             if (Ufull, Vfull) == (0, 1) then U(m,s), S(s,n), VT(n,n)
 *             if (Ufull, Vfull) == (1, 1) then U(m,m), S(m,n), VT(n,n)
 *
 * @param      U      Unitary matrix U (m x m)
 * @param      S      Sigular values array S (s x 1)
 * @param      VT     Transpose of unitary matrix V (n x n)
 * @param[in]  A      matrix to perform SVD
 * @param[in]  m      rows of A
 * @param[in]  n      columns of A
 * @param[in]  Ufull  Boolean for full U matrix
 * @param[in]  Vfull  Boolean for full V matrix
 * @param      buf   buffer
 *
 * @return     Error code
 */
int linalg_svd(double* U, double* S, double* VT,
               const double* A, integer m, integer n,
               bool Ufull, bool Vfull,
               LinalgBuffer* buf);

int main(void)
{
    double U[ROWS * ROWS];
    double S[ROWS];
    double VT[COLS * COLS];

    const double A[ROWS * COLS] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    const int m = ROWS;
    const int n = COLS;

    const bool Ufull = false;
    const bool Vfull = false;

    LinalgBuffer buf = {0, BUF_SIZE, {}};

    int result = linalg_svd(U, S, VT, A, m, n, Ufull, Vfull, &buf);
    printf("\nresult = %d,", result);
    printf("\nU[0, 0] = %f,", U[0]);
    return 0;
}

int linalg_svd(double* U, double* S, double* VT,
               const double* A, integer m, integer n,
               const bool Ufull, const bool Vfull, LinalgBuffer* buf)
{
    int retval = 0;
    /* alias data */
    const integer A_len = m * n;
    if (buf->max_len < buf->len + A_len)
    {
        // not enough buffer for copy of A
        retval = 1;
    }
    else
    {
        doublereal* Ain = &buf->data[buf->len];
        buf->len += A_len;

        char Uopt = (Ufull ? 'A' : 'S');
        char Vopt = (Vfull ? 'A' : 'S');

        memcpy(Ain, A, ((const size_t)A_len) * sizeof(double));

        integer ldvt = n;
        if (m < n && !Vfull)
        {
            ldvt = m;
        }

        doublereal work_query = 0.0;
        integer lwork = -1;
        integer info = 0;
        dgesvd_(&Uopt, &Vopt, &m, &n, Ain, &m, S, U, &m, VT, &ldvt, &work_query, &lwork, &info);
        lwork = (integer)work_query;

        if (info == 0)
        {
            if (buf->max_len < buf->len + lwork)
            {
                buf->len -= A_len;
                // not enough buffer for svd operation
                retval = 2;
            }
            else
            {
                double* work = &buf->data[buf->len];
                buf->len += lwork;

                /* perform SVD */
                dgesvd_(&Uopt, &Vopt, &m, &n, Ain, &m, S, U, &m, VT, &ldvt, work, &lwork, &info);

                if (info != 0)
                {
                    /* interpret dgesvd decomposition info here */
                    if (info < 0)
                    {
                        // Some intput error
                        retval = 3;
                    }
                    else
                    {
                        // DBDSQR did not converge (see DGESVD)
                        retval = 4;
                    }
                }

                // free lwork
                buf->len -= lwork;
            }
        }
        else
        {
            /* interpret dgesvd work query info here */
            retval = 5;
        }

        // free A
        buf->len -= A_len;
    }
    return retval;
}
