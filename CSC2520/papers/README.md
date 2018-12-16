
### alignment & registration

+ [1991_object_modeling_by_registration_of_multiple_range_images](1991_object_modeling_by_registration_of_multiple_range_images.pdf)
    + first iteration of ICP (iterative closest point)
    + goal  
        + construct 3D model from physical objects using multiple views
        + proposed approach to register successive views with enough overlapping area to accurately do transformation between views by minimizing a functional
    + general steps 
        + data acquisition 
        + registration between views
            + view is a 3D surface information of the object from a specific point of view
            + goal is to find transformation between views
                + find inter-frame transformation of range images through image registration
                + initial approximation of transformation inferred from range finder setup
                + iterative process minimizing a least square measure, requiring no point-to-point correspondence
        + integration of views
            + converting each view to a spherical or cylindrincal coordinate
    + registration 
        + 2 views of a surface is registered if they coincide, when one view is placed at a proper position and orientation relative to the other. 
            + any pair of points from 2 views representing the same surface point can be brought into coincidence by 1 rigid transformation
        + Optimize for transformation's parameter space so that D(P, Q), sum of distances for overlapping regions of points P in 1 view to corresponding overlapped regions of points Q in another view, is minimized
            + problem   
                + optimization nonlinear
                + generally do not konw what f,g are
            + solution 
                +  use an approximate transformation between two views, so that a good initial point might help with optimization procedures
        + choosing evaluation function for surface registration 
            + direct correspondence between points in 2 views is hard to obtain ...
            + instead minimize distances from points on one surface to the other.
            + but is hard to implement since finding corrsponding point q_j for p_i is an optimization itself, but since optimization is iterative, we can use T^{k-1} in k-th iteration to find the corresponding q_j for p_i at iteration k. And at iteration=1, we use T^0, the apprximate transformation.
            + additionally, use an approximation point instead of considering q as any point in Q, i.e. q is intersection of T^{k-1} transformed normal vector n_{p_i}'s intersection with Q
            + For faster convergence, consider q in line tangent to q_j, S_j, instead of all of Q. This point is given by the iterative method and the initial q is given by the approximate transformation T^0
            + In summary 
                + energy in k-th iteration is `e^k = sum(d^2(T^k * p_i, S_j^k))` where `S_j^k` is the tangent plane.
        + line-surface intersection
            + in implementing idea of registration, need to find intersection of line `l` passing through p and in direction of surface normal `n_p` of `P` at `p` with surface `Q` by finding intersection of `l` with tangent plane to `Q` in the neighborhood
            + `P = P(x,y)` and `Q = Q(x,y)`
        + registration algorithm


+ [1992_a_method_for_registration_of_3D_shapes](1992_a_method_for_registration_of_3D_shapes.pdf)
    + iterative closest point ICP algorithm
    + abstract
        + goal
            + register unfixtured rigid objects to an idealized geometric model 
        + free-form surface matching problem that solves
            + point-set matching problem
            + free-form curve matching problem
        + input representation can be any of 
            + point set
            + implicit curves/surfaces
            + parametric curves/surfaces
            + triangle mesh
            + ...

+ [2015_realtime_face_tracking_and_animation](2015_realtime_face_tracking_and_animation.pdf)
    + 3.2-3.3 registration
    + matching energy 3.2
        + goal
            + align X -> Y, Z is deformed version of X to be aligned with Y
        + matching energy
            + a measure of how close surface Z is to Y
            + intuitively how close each point z \in Z is to the surface Y
        + iterative closest point (ICP)
            + iteratively update Z, such that matching energy + prior energy is minimized 
            + speed up convergence with first-order approximation
                + i.le. use point-to-plane instead of point-to-point energy
    + prior energy 3.3 
        + priors encode (geometric) properties of object X being registered
        + examples
            + rigid ... only allow rotations/translations
                + squared sum of distance between z and rigid transformation of z
            + deformation ... geometric priors
            + linear model
            + general shape
                + 


### reconstruction


+ [2006_poisson_surface_reconstruction.pdf](2006_poisson_surface_reconstruction.pdf)
    + idea: 
        + reconstruct 3D surface from point samples
        + finding a scalar function \Chi whose gradient best approximates the vector field defined by samples (from points)
            + i.e. compute \Chi such that its laplacian approximates divergence of vector field
        + Given an implicit function f whose 
            + value is zero at point p_i 
            + gradient at p_i equates the normal vector 
        + 
    + divergence 
        + vector operator that produces a scalar field, represent volumne density of outward flux of a vector field from an infinitesimal volume around a given point
    + laplacian
        + Δf(p): rate at which the average value of f over spheres centered at p deviates from f(p) as the radius of sphere grows
    + poisson problem
        + laplace of an unknown function equating a known function. Goal is to find the unknown funciton
    + octree
        + tree structure where each node has exactly 8 children, used to represent 3D partition of a cube
    + implicit surface
        + a surface defined by F(x,y,z) = 0
        + can be represented by filling values of F to a grid of voxels
    + linear interpolation 
    + marching cubes algorithm





