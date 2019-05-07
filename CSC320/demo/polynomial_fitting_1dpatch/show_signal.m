function show_signal(signal, signalrange, x, w, position, labelt, labelx, labely)

subplot('position',position);
plot(signal);
hold on;
axis([1 length(signal) signalrange(1) signalrange(2)]);
title(labelt);
xlabel(labelx);
ylabel(labely);
plot([x-w x-w],[signalrange(1) signalrange(2)],'r');
plot([x+w x+w],[signalrange(1) signalrange(2)],'r');
plot([x x],[signalrange(1) signalrange(2)],'b');
hold off;

