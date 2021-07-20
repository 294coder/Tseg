X=load('C:\first_3\partition\26\Pointcloud\all.txt');
 %Y = subfun_rotate(X);
%  x=Y.X(:,1);
%  y=Y.X(:,2);
%  z=Y.X(:,3)
x=X(:,1);
y=X(:,2);
z=X(:,3);
figure(1)
scatter3(x,y,z,'.'),xlabel('x'),ylabel('y'),zlabel('z')
 
 %调整
theta_x= 20*pi/180;
x_fit=[1,0,0;
       0,cos(theta_x), -sin(theta_x);
       0,sin(theta_x),cos(theta_x)];

 theta_y= -6*pi/180;
 y_fit=[cos(theta_y),0,sin(theta_y);
         0,1,0;
        -sin(theta_y),0,cos(theta_y)];
    
theta_z=-54*pi/180;
z_fit=[cos(theta_z), -sin(theta_z),0;
           sin(theta_z),cos(theta_z),0;
           0,0,1];
X_min=(x_fit*y_fit*z_fit*X.').';
figure(9)
x_min=X_min(:,1);
y_min=X_min(:,2);
z_min=X_min(:,3);
c=z_min;
scatter3(x_min,y_min,z_min,50,c,'.'),xlabel('x'),ylabel('y'),zlabel('z')
figure(10)
plot(x_min,y_min,'.')





M=x_fit*y_fit*z_fit;
X_single_all=(M*X.').';
x1=X_single_all(:,1);
y1=X_single_all(:,2);
z1=X_single_all(:,3);
figure(3)
scatter3(x1,y1,z1,'.'),xlabel('x1'),ylabel('y1'),zlabel('z1')

X_cover=load('C:/first_3/26咬合接触分布测试集/26_002_cover1.txt');
M_register=[0.909 0.348 0.231;
            0.056 0.446 -0.893;
            -0.414 0.825 0.386;] ;   
X_cover1=(M_register*X_cover.').';
X_cover1(:,1)=X_cover1(:,1)+7.298;
X_cover1(:,2)=X_cover1(:,2)+3.786;
X_cover1(:,3)=X_cover1(:,3)+25.447;
X_cover1=(M*X_cover1.').';
x_cover1=X_cover1(:,1);
y_cover1=X_cover1(:,2);


path1='C:\first_3\26咬合接触分布测试集\cs002-lu\26.txt';
path2='C:\first_3\fq001-lu\26\Polylines_change\';
[XV,YV,ZV]=original(path1,path2);
% figure(4)
% plot3(XV{1},YV{1},ZV{1},XV{2},YV{2},ZV{2},XV{3},YV{3},ZV{3},XV{4},YV{4},ZV{4},XV{5},YV{5},ZV{5},XV{6},YV{6},ZV{6},X_cover(:,1),X_cover(:,2),X_cover(:,3),'r.')

[mark,XV,YV]=partitions(M,x_cover1,y_cover1);
figure(5)
plot(XV{1},YV{1},XV{2},YV{2},XV{3},YV{3},XV{4},YV{4},XV{5},YV{5},XV{6},YV{6},x_cover1,y_cover1,'.')