#' Matrix and arithmetic operations for big.matrix objects
#' 
#' These methods extend the base matrix multiplication operator
#' (\code{\link[base]{%*%}}) and the group generic \code{\link[base]{Arithmetic}} so
#' that \code{\link[bigmemory]{big.matrix}} objects can interoperate with base
#' R matrices and numeric scalars using the high-performance routines provided
#' by \pkg{bigalgebra}.
#' 
#' Matrix multiplications dispatch to \code{bigalgebra::dgemm()}, mixed
#' arithmetic on matrices relies on \code{bigalgebra::daxpy()}, and
#' scalar/matrix combinations use \code{bigalgebra::dadd()} when appropriate.
#' 
#' @name bigmatrix-operations
#' @rdname bigmatrix-operations
#' 
#' @aliases bigmatrix-operations 
#' @aliases %*%,big.matrix,big.matrix-method
#' @aliases %*%,matrix,big.matrix-method
#' @aliases %*%,big.matrix,matrix-method
#' @aliases Arith,big.matrix,big.matrix-method
#' @aliases Arith,big.matrix,matrix-method
#' @aliases Arith,matrix,big.matrix-method
#' @aliases Arith,numeric,big.matrix-method
#' @aliases Arith,big.matrix,numeric-method
#' 
#' 
#' @docType methods
#' 
#' @param x,y Matrix operands supplied either as \code{big.matrix} instances or
#' base R matrices, depending on the method signature.
#' 
#' @param e1,e2 Numeric operands, which may be \code{big.matrix} objects, base
#' R matrices, or numeric scalars depending on the method signature.
#' 
#' @seealso [bigmemory::big.matrix()], [bigalgebra::dgemm()],
#'   [bigalgebra::daxpy()], [bigalgebra::dadd()]
#' 
#' @keywords methods
#' @export
#' 
#' @examples 
#' if (requireNamespace("bigmemory", quietly = TRUE) &&
#'     requireNamespace("bigalgebra", quietly = TRUE)) {
#'   x <- bigmemory::big.matrix(2, 2, init = 1)
#'   y <- bigmemory::big.matrix(2, 2, init = 2)
#'   x %*% y
#'   x + y
#'   x * 3
#' }
#' 
setMethod("%*%",signature(x="big.matrix", y="big.matrix"),
          function(x,y)
          {
            dgemm(A=x, B=y)
          },
          valueClass="big.matrix"
)

#' @rdname bigmatrix-operations
#' @export
setMethod("%*%",signature(x="matrix", y="big.matrix"),
          function(x,y)
          {
            if(dim(x)[2] != dim(y)[1]) stop("non-conformant matrices")
            R = options("bigalgebra.mixed_airthmetic_returns_R_matrix")[[1]]
            if(!is.null(R) && R) return(dgemm(A=x, B=y, C=0))
            dgemm(A=x, B=y)
          },
          valueClass="matrix"
)

#' @rdname bigmatrix-operations
#' @export
setMethod("%*%",signature(x="big.matrix", y="matrix"),
          function(x,y)
          {
            if(dim(x)[2] != dim(y)[1]) stop("non-conformant matrices")
            R = options("bigalgebra.mixed_airthmetic_returns_R_matrix")[[1]]
            if(!is.null(R) && R) return(dgemm(A=x, B=y, C=0))
            dgemm(A=x, B=y)
          },
          valueClass="matrix"
)

#' @rdname bigmatrix-operations
#' @export
setMethod("Arith",c(e1="big.matrix", e2="big.matrix"),
          function(e1,e2)
          {
            op = .Generic[[1]]
            switch(op,
                   `+` = daxpy(1.0,e1,e2),
                   `-` = daxpy(-1.0,e2,e1),
                   stop("Undefined operation")
            )
          }
)

#' @rdname bigmatrix-operations
#' @export
setMethod("Arith",c(e1="big.matrix", e2="matrix"),
          function(e1,e2)
          {
            op = .Generic[[1]]
            switch(op,
                   `+` = daxpy(1.0,e1,e2),
                   `-` = daxpy(-1.0,e2,e1),
                   stop("Undefined operation")
            )
          }
)

#' @rdname bigmatrix-operations
#' @export
setMethod("Arith",c(e1="matrix", e2="big.matrix"),
          function(e1,e2)
          {
            op = .Generic[[1]]
            switch(op,
                   `+` = daxpy(1.0,e1,e2),
                   `-` = daxpy(-1.0,e2,e1),
                   stop("Undefined operation")
            )
          }
)

#' @rdname bigmatrix-operations
#' @export
setMethod("Arith",c(e1="numeric", e2="big.matrix"),
          function(e1,e2)
          {
            op = .Generic[[1]]
            if(length(e1)==1) {
              if (op=="*") 
                return(daxpy(e1,e2))
              return(switch(op,
                            `+` = dadd(e2, e1, 1.0, 1),
                            `-` = dadd(e2, e1,-1.0, 1),
                            stop("Undefined operation")
              ))
            }
            stop("e1 is not a scalar")
          }
)

#' @rdname bigmatrix-operations
#' @export
setMethod("Arith",c(e1="big.matrix", e2="numeric"),
          function(e1,e2)
          {
            op = .Generic[[1]]
            if(length(e2)==1) {
              if( op=="*") 
                return(daxpy(e2,e1))
              return(switch(op,
                            `+` = dadd(e1,e2, 1.0, 0),
                            `-` = dadd(e1,e2, -1.0, 0),
                            stop("Undefined operation")
              ))
            }
            stop("e2 is not a scalar")
          }
)
