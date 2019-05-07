


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


+ [2004_exemplar_based_image_inpainting](2004_region_filling_and_object_removal_by_exemplar_based_image_inpainting.pdf)
    + absract
        + noval algo for removing objects from digital photographs and replace them with visually plausible backgrounds
        + previous
            + texture synthesis with stochasticity
                + exemplar-based: generate new texture by copying new color from the source
            + image inpainting 
                + fill holes by propagating linear structures into target region via diffusion (heat flow pde ...) introduce blurring ...
    + exemplar-based synthesis
        + how it works
            + find candidate match along boundy of two textures
            + best matching patch copied to partially fill target location (contains both texture and structure)
        + filling order is critical
            + should give higher priority of synthesis of target regions which lie on the continuation of image structures (edges)
    + proposed region-filling algorithm
        + inputs
            + target region `\Sigma`
            + template window size (defaults: 9x9)
        + data structure
            + each pixel has _colour_ value and a _confidence_ value
            + patch front gives _priority_ value, determines order in which they are filled
        + patch priority
            + biased towrads patches no continuation of strong edges and are surrounded by high-confidence pixels
            + `P = CD`
                + `P` is priority at `p`
                + `C` is a measure of confidence of reliable information surrounding pixel `p`, additional preference given to pixels that are filled early on or never part of the target region; approximately enforces concentric fill order
                + `D` strength of isophotes hitting the front `\partial \Omega` at each iteration; encourages linear structures to be filled first
            + pixel on boundary with highest priority is considered first, until target is filled
        + texture/structure propagation
            + find _exemplar_ patch in source region that is most similar to the template region, value of each pixel-to-be-filled is copied from corresponding location inside the _exemplar_ patch
        + confidence values update
            + filled pixels inherit pixels of the then-best pixel confidence value
            + confidence values decay during each iteration