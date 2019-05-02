

+ triangle mesh https://en.wikipedia.org/wiki/Triangle_mesh
    + triangle strip https://en.wikipedia.org/wiki/Triangle_strip
        + connected triangles sharing vertices
        + memory efficient, 3N -> N+2, where N is number of vertices
            + faster to load into RAM
        + even numbered triangle (starting from 1) would be reversed resulting in the original triangles
    + data structure
        + inserting triangle
        + removing triangle

+ voxelization
    + converting geometric objects from their continuous geometric representation into a set of voxels that best approximates the continuous object.