# Null-space-Rayleigh-Quotient-Iteration
Implementation of a null-space Rayleigh Quotient Iteration algorithm in Jax with sparse matrices. Works on the GPU! Could be much faster if we use a better way to solve the linear subproblem.

Solves a problem that looks like:

![equation](http://www.sciweavers.org/tex2img.php?eq=%5Cmin_XX%5E%5Ctop%20AX%20%5Cquad%20%5Ctext%7Bs.t.%20%7DX%5E%5Ctop%20X%20%3D%201%2C%20%5Cquad%20v%20X%20%3D%20c&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

It can be solved as a generalized eigenvalue problem of PAP, with P the projection onto the subspace orthogonal to v. One issue is that if A is sparse, PAP could be dense and we could run into memory or runtime problems with big matrices.
