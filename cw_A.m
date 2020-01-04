%------------------------------------A1----------------------------------%
% Compute staionary point
syms x y
v (x,y) = 100 * (y - x.^2).^2 + (1-x).^2;
vx = diff (v, x)
vy = diff (v, y)
[x0, y0]= solve(vx, vy)
% Verify minimizer
vxx(x,y) = diff (vx, x);
vxy(x,y) = diff (vx, y);
vyx(x,y) = diff (vy, x);
vyy(x,y) = diff (vy, y);
v2 = [vxx(x0,y0), vxy(x0,y0); vyx(x0,y0), vyy(x0,y0)]
[u, s] = eig (v2)
%------------------------------------A2----------------------------------%
% Plot level sets
x = linspace(-2,2); y = linspace(-1.5,3);
[x,y] = meshgrid(x,y); 
v = v (x,y);
levels1 = [0.1,0.25,0.5,1,3,5,10,25,50,100];
levels2 = [200: 100: 1000];
% levels3 = [2000: 1000: 10000];
levels = [levels1 levels2];
figure(1)
[C, h] = contour(x,y,v,levels,'linewidth', 1.5), colorbar
clabel(C,h);
xlabel('x');
ylabel('y');
title ("The Level sets of Rosenbrock function");
axis([-2 2 -1.5 3]);
axis square;
grid on;
% hold on
% %------------------------------------A3----------------------------------%
% tolerance = 1e-5;
% % line search setting
% alpha = 0.2;
% gamma = 0.45;
% iter = 0 ;
% tbar=1;
% x0 = [-0.75; 1];
% k = 0;
% % initialize X
% x = x0 ;
% history(:,1) = x0;
% [fun, g] = f(x);
% % Gradient-Armijo
% while norm(g) > tolerance
%     fprintf('DEBUG: %i %f\n',iter,norm(g))
%     d = -g;
%     t = tbar;
%     % Armijo
%     while f(x+t*d) > fun + alpha*g'*d*t
%         t = gamma*t ;
%     end
%     fprintf('DEBUG: %i %f\n',iter,norm(g))
%     x = x + t*d ;
%     [fun, g] = f(x);
%     iter=iter+1;
%     k = k + 1;
%     history(:,k+1) = x;
% end
% title('Gradient Method with Armijo')
% plot(history(1,:),history(2,:),'r.-')
% function [fun, g] = f(x)
%     fun =  100 * (x(2) - x(1).^2).^2 + (1-x(1)).^2;;
%     g = [  2*x(1) - 400*x(1)*(- x(1)^2 + x(2)) - 2 ; - 200*x(1)^2 + 200*x(2) ];
% end
% %------------------------------------A3_----------------------------------%
