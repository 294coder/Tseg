function [XV,YV,ZV]=original(path1,path2)
 X=load(path1);
 x=X(:,1);
 y=X(:,2);
 z=X(:,3);
 base=path2;
 XV={};
 YV={};
 ZV={};
 for k=1:6
    clear xv
    clear yv
    read_path=strcat(base,num2str(k),'.txt');
    P= load(read_path);
    xv=P(:,1);
    yv=P(:,2);
    zv=P(:,3);
    XV{k}=xv;
    YV{k}=yv;
    ZV{k}=zv;
 end
end

  
 

 
 
