pos = 150;
rowcol = 2;
W = 10;
n = 1;
deriv = 1; 
imfname = 'window.pgm';
wls = 1;
fignum = 1;
skip = 0;

qq = input(['skip row display? (default=',num2str(skip),') ']);
if (length(qq) > 0)
    skip = qq;
end

qq = input(['enter figure number (default=',num2str(fignum),') ']);
if (length(qq) > 0)
    fignum = qq;
end


qq = input(['enter filename (default=',imfname,') ']);
if (length(qq) > 0)
    imfname = qq;
end


qq = input(['enter 1 for row 2 for col (default=',num2str(rowcol),') ']);
if (length(qq) > 0)
    rowcol = qq;
end

qq = input(['enter position (default=',num2str(pos),') ']);
if (length(qq) > 0)
    pos = qq;
end

qq = input(['enter polynomial degree (default=',num2str(n),') ']);
if (length(qq) > 0)
    n = qq;
end

qq = input(['enter derivative to show (default=',num2str(deriv),') ']);
if (length(qq) > 0)
    deriv = qq;
end

qq = input(['enter 1 for LS and 2 for WLS (default=',num2str(wls),') ']);
if (length(qq) > 0)
    wls = qq;
end

if (wls == 1)
    polyfit_demo(fignum, imfname, pos, rowcol, W, n, deriv, 'polyfitLS', skip);
elseif (wls == 2)
    polyfit_demo(fignum, imfname, pos, rowcol, W, n, deriv, 'polyfitWLS', skip);
else 
    polyfit_demo(fignum, imfname, pos, rowcol, W, n, deriv, 'polyfitRAN', skip);
end
