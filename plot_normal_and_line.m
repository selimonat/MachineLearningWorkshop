function plot_normal_and_line(normal_w)


if length(normal_w) == 2
    normal_w(3) = 0;
end
%
plot_vectors(normal_w(1:2)); %plot the vector
hold on;
%
w           = normal_w(1:2);             % this is the normal vector.
w0          = normal_w(3);               % this is the bias term, that shifts the line up or down.
w_line(1:2) = [-normal_w(2) normal_w(1)];% this is the 90 degree rotated vector, the vector that goes along the line

line        = w_line(:)*linspace(min(xlim)/max(abs(w_line)),max(xlim)/max(abs(w_line)),10);

line        = line + repmat(w'/norm(w)*w0,1,10);
plot(line(1,:),line(2,:),'o-')
axis tight;axis square;
set(gca,'xlim',[min(xlim) max(xlim)],'ylim',[min(ylim) max(ylim)]);
%
