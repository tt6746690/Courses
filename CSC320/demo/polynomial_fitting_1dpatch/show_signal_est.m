function show_signal_est(signal, signalrange, signal_est_x, signal_est, x, w, position, labelt, labelx, labely)

subplot('position', position);
plot(signal,'b');
hold on;
axis([1 length(signal) signalrange(1) signalrange(2)]);
title(labelt);
xlabel(labelx);
ylabel(labely);

plot(signal_est_x,signal_est,'r');
plot(signal_est_x(length(signal_est_x)),signal_est(length(signal_est_x)),'ro');
plot([x(length(x)) x(length(x))],[signalrange(1) signalrange(2)],'r');
hold off;
