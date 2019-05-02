


+ [1997_recovering_HDR](1997_recovering_high_dynamic_range_radiance_maps_from_photographs.pdf)
    + abstract
        + given differently exposed photographs, combine them to recover (nonlinear) response function of imaging process
        + fuse multiple images into a single, high dynamic range radiance map whose pixel values are proportional to the true radiance values in the scene
    + intro 
        + `Z = f(X)` where 
            + `X` is exposure
            + `Z` is response (digital number)
            + `f` is a monotonically increasing nonlinear function (hence invertible) of the developing, scanning, digitizing process
        + reciprocity assumption 
            + characteristic (optical density `D` vs. log of exposure `X`) curve
            + `X = E\dt`
                + `E` irradiance
                + `\dt` exposure time
            + reciprocity
                + only the product is important, i.e. halving `\dt` while doubling `E` wil not change the resulting optical density
            + reciprocity failure 
                + reciprocity breaks down for large/small exposure time
    + assumptions
        + irradiance `E` linear to radiance `L`
        + irradiance `E_i` constant for each pixel `i`
        + `Z_ij` for `i` pixel and `j`-th picture


+ [1996_blue_screen_matting](1996_blue_screen_matting.pdf)
    + definitions
        + matting problem 
            + separate forground image from a rectangular background image
        + matte
            + strip of monochrome film that is transparent over area of interests and opaque elsewhere
        + holdout matte (similar to how alpha channel functions)
            + complement of matte, opaque in parts of interests
        + matte is successfully pulled (inverse problem of geometry-based rendering of both object and its matte)
    + problem 
        + extract a matte for a foreground, given only a composite image containing it