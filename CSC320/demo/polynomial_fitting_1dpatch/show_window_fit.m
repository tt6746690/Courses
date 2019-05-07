function show_window_fit(signal_window, signalrange, window_fit, x, w, position, labelt, labelx, labely)

subplot('position',position);
plot([-w:w],signal_window,'ro');
hold on;
title(labelt);
xlabel(labelx);
ylabel(labely);
axis([-w w signalrange(1) signalrange(2)]);
plot([-w:w], window_fit, 'b');
hold off;
