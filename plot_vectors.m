function plot_vectors(v,varargin)

for i = 1:size(v,1)   
    plot([0,v(i,1)],[0,v(i,2)],'ko-','linewidth',2,varargin{:});
    hold on;
end
axis tight;
box off;
grid on;
axis square;
axis equal;
Xlim = ceil(max(abs([v(:)]))*1.2);
Ylim = Xlim;
set(gca,'xlim',[-Xlim Xlim],'ylim',[-Ylim Ylim]);
hold off;