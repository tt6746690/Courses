function show_photo(photo, p1, p2, liner, linec, position, labelt)

winpos = [p1' p2']';
subplot('position', position);
colormap(gray);
imagesc(abs(double(photo))/4);
title(labelt);
hold on;
plot(liner,linec,'r');
plot(winpos(:,1),winpos(:,2),'y');
hold off;
