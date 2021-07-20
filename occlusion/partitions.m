function [mark,XV,YV]=partitions(M,x,y)
%base='C:\first_3\fq001-lu\26\Polylines_change\';
base='C:\first_3\partition\26\Polylines\';
XV={};
YV={};
mark=zeros(1,6);
for k=1:6
    clear xv
    clear yv
    read_path=strcat(base,num2str(k),'.txt');
    P= load(read_path);
%     P_rorate=P;
    P_rorate= (M*P.').';
    xv=P_rorate(:,1);
    yv=P_rorate(:,2);
    xv= [xv ; xv(1)];
    yv= [yv ; yv(1)];
%     figure(k+2)
%     plot(xv,yv)
    in = inpolygon(x,y,xv,yv);
    mark(k)=any(in);
    XV{k}=xv;
    YV{k}=yv;
end
