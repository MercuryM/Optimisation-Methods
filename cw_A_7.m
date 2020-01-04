clear
clc
close all
%% ------------------------------------A1----------------------------------%
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
%% ------------------------------------A2----------------------------------%
% Plot level sets
x = linspace(-2,2); y = linspace(-1.5,3);
[x,y] = meshgrid(x,y); 
v = v (x,y);
levels1 = [0.1,0.25,0.5,1,3,5,10,25,50,100];
levels2 = [200: 100: 1000];
% levels3 = [2000: 1000: 10000];
levels = [levels1 levels2];
figure(1)
contour(x,y,v,levels,'linewidth', 1.5), colorbar
xlabel('x');
ylabel('y');
title ("The Level sets of function v(x,y)");
axis([-2 2 -1.5 3]);
grid on
axis square;
hold on
%% ------------------------------------A7----------------------------------%
% Rosenbrock function
[x,y]=meshgrid(-2:0.01:3,-2:0.01:5);
v=100 * (y - x.^2).^2 + (1 - x).^2;
x0 = [-3/4; 1];
[x,val,k] = simplex(x0)
function [x,val,k] = simplex(x0)
k = 0;maxk = 5000;
f0 = fun(x0);
stepsize = 0.5;
x1 = [x0(1)+stepsize;x0(2)];
x2 = [x0(1)+stepsize/2; x0(2)-sqrt(3)/2*stepsize];
p=[x0 x1 x2];
history(:,1)=x0;
history(:,2)=x1;
history(:,3)=x2;
jk = log((x0(1)-1)^2 + (x0(2)-1)^2);
recordjk(:,1)=jk;
value(:,1) =fun(x0);
e = 10e-9;
a = 1.95;
while k < maxk
    avef = (fun(p(:,1))+ fun(p(:,2))+ fun(p(:,3)))/3;
    cri=((fun(p(:,1))-avef)^2 + (fun(p(:,2))-avef)^2 + (fun(p(:,3))-avef)^2)/3;
    if (cri < e), break; end
    %sort the max out
    F=[fun(p(:,1));fun(p(:,2));fun(p(:,3))];
    [F_sort,ind]=sort(F);
    F_max=F(ind(3));
    xmax=ind(3);
    xk = (p(:,1)+p(:,2)+p(:,3))/3;
    p(:,xmax) = xk + a*(xk-p(:,xmax));
    k = k + 1;
    history(:,k + 3) = p(:,xmax);
    jk=log((p(1,xmax)-1)^2 + (p(2,xmax)-1)^2);
    recordjk(:,k + 1)=jk;
    k
    xop= p(:,xmax);
    opval=fun(p(:,xmax));
    value(:,k+1)=opval;
end
x = xop;
val = opval;
% plot the sequence of points
% K = linspace(0, k, k + 1);
plot(history(1,:),history(2,:),'r*-')
text(x0(1),x0(2),['Initial point'],'color',[1 0.5 0])
text(xk(1),xk(2),['Terminate point'],'color',[1 0.5 0])
xlabel('x')
ylabel('y')
title('Simplex Method')
% plot the cost jk-k
figure(2)
plot(recordjk(1,:))
xlabel('k')
ylabel('J')
xlim([1,inf])
title('Cost of Simplex Method')
figure(3)
plot(value(1,:))
xlabel('k')
ylabel('v')
ylim([-5,inf])
title('Value of Function versus k')
end
function f = fun(x)
f = 100*(x(1)^2-x(2))^2 + (x(1)-1)^2;
end



