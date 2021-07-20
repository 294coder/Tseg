function [Rotate] = subfun_rotate(X,A)
    if nargin >1
        angle = A;
    else
        angle = (-180:5:180)*pi/180;
    end
        

   R_x = @(theta_x)[1,0,0;
           0,cos(theta_x), -sin(theta_x);
           0,sin(theta_x),cos(theta_x)];
   R_y = @(theta_y)[cos(theta_y),0,sin(theta_y);
           0,1,0;
           -sin(theta_y),0,cos(theta_y)];
       
   R_z = @(theta_z)[cos(theta_z), -sin(theta_z),0;
           sin(theta_z),cos(theta_z),0;
           0,0,1];
       
%    Rotate_X = R_x(rotate_A(1))* R_y(rotate_A(2))*R_z(rotate_A(3))* X';
%    Rotate_X = Rotate_X.';
   
   N = length(angle);
   Max_S = 0;
   
   for i1 = 1:N
       for i2 = 1:N
           for i3 =1:N
               
                temp_M = R_x(angle(i1))* R_y(angle(i2))*R_z(angle(i3));
                temp_X = temp_M* X';
                temp_X = temp_X.';
  
               minmax_x = [min(temp_X(:,1)),max(temp_X(:,1))];
               minmax_y = [min(temp_X(:,2)),max(temp_X(:,2))];
               s = (minmax_x(2)-minmax_x(1))*(minmax_y(2)-minmax_y(1));
               if s > Max_S
                   Max_S = s;
                   Rotate.A = [angle(i1),angle(i2),angle(i3)];
                   Rotate.X = temp_X;
                   Rotate.M = temp_M;
               end
           end
       end
   end
   
   if 1
       y = 45*pi/180;
       z = 30*pi/180;
        Rotate.X = (R_y(y)*R_z(z)*Rotate.X.').';
        Rotate.A = Rotate.A + [0,y,z];
        Rotate.M = R_y(y)*R_z(z)*Rotate.M;
  end
   %figure
   %pcshow(Rotate_X);
end

