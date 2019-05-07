function polyfit_demo(fignum, imfname, pos, rowcol, W, n, deriv, funstr, skip)
% function polyfit_demo(fignum, im, pos, rowcol, W, n, deriv)
%

im = double(imread(imfname));

textheight = 0.05;
graphheight = 0.15;

figure(fignum);
clf;

if (rowcol == 1)
    signal = im(pos,:);
    liner = [1 size(im,2)];
    linec = [pos pos];
    maxpos = size(im,2)-W;
    maxrange = size(im,1);
    rowcolstr = 'row';
    colrowstr = 'col';
    vec = [0 1];
    tvec = [1 0];
else 
    signal = im(:,pos)';
    liner = [pos pos];
    linec = [1 size(im,1)];
    maxpos = size(im,1)-W;
    maxrange = size(im,2);
    rowcolstr = 'col';
    colrowstr = 'row';
    vec = [1 0];
    tvec = [0 1];
end

x = [];
I = [];

if (skip == 0)

for center = (W+1):maxpos

    patch = signal((center-W):(center+W));
    [d v] = polyfitWLS(patch',n);
    x = [x center];
    I = [I d];

end

maxvals = max(I')';
minvals = min(I')';

x = [];
I = [];

for center = (W+1):maxpos

    patch = signal((center-W):(center+W));
    [d v] = eval([funstr,'(patch'',n);']);
    %[d v] = polyfitWLS(patch',n);
    x = [x center];
    I = [I d];

    show_photo(im, vec*pos+(center-W)*tvec, vec*pos+(center+W)*tvec, liner, linec, [0.05 0.45 0.20 0.5],...
        'Input photograph');

    graphpos = graphheight + textheight;

    show_signal(signal, [minvals(1) maxvals(1)], center, W, [0.35 1-graphpos 0.60 graphheight], ...
        ['Image intensities in ',rowcolstr,' ',num2str(pos)],...
        'pixel position (column #)',...
        'pixel intensity');
    
    graphpos = graphpos + 2*textheight + graphheight;

    show_window_fit(patch, [minvals(1) maxvals(1)], v, center, W, [0.35 1-graphpos 0.60 graphheight], ...
        ['Least-squares fit of intensities in patch of radius w=',num2str(W),' centered at ',colrowstr,' ',num2str(center)],...
         'pixel position within patch (x)',...
         'intensity I(x)')
    
    graphpos = graphpos + 2*textheight + graphheight;

    show_signal_est(signal, [minvals(1) maxvals(1)], x, I(1,:), center, W, [0.35 1-graphpos 0.60 graphheight],...
        ['Estimated intensity at patch center (I(0))'],...
         'pixel position (column #)',...,
         'pixel intensity')
    
    graphpos = graphpos + 2*textheight + graphheight;

    if (deriv == 1) 
        show_signal_est_only(signal, [minvals(deriv+1) maxvals(deriv+1)], x, I(deriv+1,:), center, W, [0.35 1-graphpos 0.60 graphheight],...
         ['Estimated 1st intensity derivative at patch center (dI(0)/dx)'],...
             'pixel position (column #)',...,
             'pixel intensity');
     else
       show_signal_est_only(signal, [minvals(deriv+1) maxvals(deriv+1)], x, I(deriv+1,:),...
           center, W, [0.35 1-graphpos 0.60 graphheight],...
           ['Estimated 2nd intensity derivative at patch center (dI2(0)/dx2)'],...
             ['pixel position'],...,
             'pixel intensity');
     end
     
     
    pause

end

end

if (rowcol == 1)
    im_d = zeros(size(im));
    im_est = zeros(size(im));
    for pos1=1:size(im,1)    
        signal = im(pos1,:);
        x = [];
        I = [];
        for center = (W+1):maxpos
            patch = signal((center-W):(center+W));
            [d v] = polyfitWLS(patch',n);
            x = [x center];
            I = [I d];
        end
        im_est(pos1,(W+1):maxpos) = I(1,:);
        im_d(pos1,(W+1):maxpos) = I(deriv+1,:);
    end
else
    im_d = zeros(size(im));
    im_est = zeros(size(im));
    for pos1=1:size(im,2)    
        signal = im(:,pos1)';
        x = [];
        I = [];
        for center = (W+1):maxpos
            patch = signal((center-W):(center+W));
            [d v] = polyfitWLS(patch',n);
            x = [x center];
            I = [I d];
        end
        im_est((W+1):maxpos,pos1) = I(1,:)';
        im_d((W+1):maxpos,pos1) = I(deriv+1,:)';
    end
end


figure(fignum+1);

show_photo(im, vec*pos+(center-W)*tvec, vec*pos+(center+W)*tvec, liner, linec, [0.05 0.05 0.25 0.95],...
        'Original photograph');

show_photo(im_est, vec*pos+(center-W)*tvec, vec*pos+(center+W)*tvec, liner, linec, [0.35 0.05 0.25 0.95],...
        'Estimated photograph');

show_photo(im_d, vec*pos+(center-W)*tvec, vec*pos+(center+W)*tvec, liner, linec, [0.65 0.05 0.25 0.95],...
        'Estimated derivative');
