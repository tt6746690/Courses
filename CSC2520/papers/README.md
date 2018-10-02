
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




### Final Project

+ [2013_robust_inside_outside_segmentation_using_generalized_winding_numbers](2013_robust_inside_outside_segmentation_using_generalized_winding_numbers.pdf)
    + algorithm for 
        + boundary/surface representation (triangle mesh) -> volumetric representation (tetrahedral mesh, i.e. voxels)
        + i.e. discretization of input's inner volume
        + deals with geometric/topological artifacts
            + self-intersection 
            + open boundaries 
            + non-manifold edges
    + Delaunay triangulation (CDT)
        + for a given set P of discrete points in a plane is a triangulation DT(P) such that no point in P is inside the circumcircle of any triangle in DT(P).
        + then segment to inside and outside volume
    + comparison
        + winding number
            + signed length of the projection of a curve onto a circle at a given point divided by 2pi
                + outside, projection cancels out, -> 0
                + inside, projection = 1
            + sharp jump at the surface, smooth elsewhere
            + a piecewise-constant segmentation of space when input is perfect (i.e. watertight)
            + if not perfect, then function well=behaved and will guide downstream E-minimization algorithms
        + signed distance field
            + do not encode segmentation confidence! smooth when crossing the surface
    + steps
        + construct inside-outside confidence function which generate the winding number
        + evaluate integral average of this function at each element in a CDT containing (V,F)
        + select a subset \Epsilon of CDF via graphcut energy optimization enforcing facet interpolation and manifoldness
    + winding number
        + is number of full revolution an observer at p tracking a moving point along C took
        + 2D
            + signed length of projection of a curve C onto the unit circle around p divided by 2pi
                + 0 -> outside
                + 1 -> inside
            + full ccw rotation  ->  w(p) = 1
            + full cw rotation -> w(p) = -1
        + 3D genearlization 
            + solid angle is the signed surface area of projection of a surface S to a point in R^3
                + angle -> solid angle
            + w(p)=  solid_angle(p) / 4pi
        + immediate discretiztion, i.e. 
            + if C is piecewise linear
            + if S is triangulated piecewise linear surface
    + open, nonmanifold, ...
        + winding number is harmonic except on C/S, implying C^{infty} smoothness and minimal oscillation
        + jump +-1 cross boundary is a confidence measure, 
    + hierarchical evaluation
        + computing w(p): simply sum the contribution of each triangle in a mesh, easily parallezable
    + segmentation & energy minimization with graphcut
        + have w(p), select a subset of CDT of convex hull of (V, F)
    + future direction 
        + winding nubmer rely on orientation of input facets, triangle soups / or with erroneous orientation need further processfing
    + conclusion    
        + respects self-intersection and correctly identifies regions of overlap in presense of artifacts such as holes



+ [2016_thingi10k_a_dataset_of_10000_3D_printing_models](2016_thingi10k_a_dataset_of_10000_3D_printing_models.pdf)
    + 10K good quality 3D printing model with annotation and is queriable


+ [2018_fast_winding_numbers_for_soups_and_clouds](2018_fast_winding_numbers_for_soups_and_clouds.pdf)
    + abstract
        + generalize winding number to point clouds
            + i.e. determine if a point is inside or outside
        + improve runtime speed for triangle soups and point clouds
            + tree based algorithm to reduce complexity, O(log m) amortized
    + application 
        + with Thingi10k + winding number -> signed voxel grid for GAN training and shape generation
    + what is bad about 2013 paper
        + direct summation of contribution of each triangle mesh is too slow
            + sublinear complexity with divide and conquer
            + GOOD in general, but needs pre-computation
            + also in case of triangle clouds, degenerates into slow direct sum
    + oriented point clouds
        + list of points P + normal vector N or m oriented triangles
        + outputs a function w, which computes winding number of any points
        + generalization 
            + mesh: solid angle computed via inverse tangent function



