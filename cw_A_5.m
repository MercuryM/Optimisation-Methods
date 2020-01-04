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
contour(x,y,v,levels,'linewidth', 1.5), colorbar
xlabel('x');
ylabel('y');
title ("The Level sets of function v(x,y)");
axis([-2 2 -1.5 3]);
grid on
axis square;
hold on
%-----------------------------------A5------------------------------------%
% x0 = [-0.75; 1];
% [x,val,k] = Polak_Ribiere(x0)
% function [x,val,k] = Polak_Ribiere(x0)
% %功能: 用pr法求解无约束问题:  min f(x)
% %输入:  x0是初始点, fun, gfun分别是目标函数和梯度
% %输出:  x, val分别是近似最优点和最优值,  k是迭代次数.
% maxk = 5000;   %最大迭代次数
% rho = 0.1; sigma = 0.1;
% k = 1;  epsilon = 1e-5;
% xk_1 = x0;
% history(:,1) = x0;
% jk = log((x0(1)-1)^2 + (x0(2)-1)^2);
% recordjk(:,1) = jk;
% value(:,1) =fun(x0);
% g = gfun(x0); d = -g;
% while(k < maxk)
%     g = gfun(x0);  %计算梯度
%     gk_1 = gfun(xk_1);
%     fprintf('DEBUG: %i %f\n',k,norm(g))
%     if(norm(d) < epsilon), break; end
%     dk_1 = d;
%     d = -g + ((g' * (g - gk_1) * dk_1)/(norm(gk_1))^2);
%     m = 0; mk = 0;
%     while(m < 100)   %Armijo搜索
%         if(fun(x0 + rho^m*d)) < (fun(x0) + sigma * rho^m * g' * d)
%             mk = m; break;
%         end
%         m = m+1;
%     end
%     xk_1 = x0;
%     x0 = x0 + rho^mk * d;
%     k = k + 1;
%     jk=log((x0(1)-1)^2 + (x0(2)-1)^2);
%     recordjk(:,k) = jk;
%     history(:,k) = x0;
%     value(:,k) = fun(x0);
% end
x0 = [-0.75; 1];
[x,val,k] = Polak_Ribiere(x0)
function [x,val,k] = Polak_Ribiere(x0)
k = 0; kmax = 5000;
xk = x0;
alpha = 1; sigma = 0.5; gamma = 0.4; epsilon = 1e-5;
history(:,k + 1) = x0;
jk = log((x0(1)-1)^2 + (x0(2)-1)^2);
recordjk(:,k + 1) = jk;
value(:,1) =fun(x0);
g = gfun(x0);
while k <= kmax
    if k == 0
        dk = -g;
    else
        dk = -g +((g' * (g - g_pre))/(norm(g_pre))^2) * dk_pre;
    end
    fprintf('DEBUG: %i %f\n',k,norm(g))
    if (norm (g) < epsilon) break; end
    dk_pre = dk;
    g_pre = g;
    alphak = armijo (xk, alpha, sigma, gamma, dk);
    xk = xk + alphak * dk;
    g = gfun(xk);
    k = k + 1;
    jk=log((xk(1)-1)^2 + (xk(2)-1)^2);
    recordjk(:,k + 1) = jk;
    history(:,k + 1) = xk;
    value(:,k + 1) = fun(xk);
end
x = xk;
val = fun(x); 
% K = linspace(0, k, k + 1);
title('Polak-Ribiere Method with Armijo')
plot(history(1,:),history(2,:),'r*-')
text(x0(1),x0(2),['Initial point'],'color',[1 0.5 0])
text(xk(1),xk(2),['Terminate point'],'color',[1 0.5 0])
figure(2)
plot(recordjk(1,:))
xlabel('k')
ylabel('J')
xlim([1,inf])
title('Cost of Polak-Ribiere with Armijo')
figure(3)
plot(value(1,:))
xlabel('k')
ylabel('v')
ylim([-5,inf])
title('Value of Function versus k')
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
function alphak = armijo(xk,alpha,sigma,gamma,dk)
% Armijo line search
g = gfun (xk);
x = xk + alpha * dk;
while fun(x)-fun(xk) > gamma * alpha * gfun(xk)' * dk
        alpha = sigma * alpha;
        x = xk + alpha * dk;
end
alphak = alpha;
end


function f = fun(x)
f = 100*(x(2)-x(1).^2).^2 + (1 - x(1)).^2;
end
%-------------------------------
function gf = gfun(x)
gf = [400*x(1)*(x(1)^2-x(2))+2*(x(1)-1), -200*(x(1)^2 - x(2))]';
end
%-------------------------