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
x0 = [-0.75; 1];
[x1,val1,k1,value1,recordjk1] = grad(x0);
[x2,val2,k2,value2,recordjk2] = newton(x0);
[x3,val3,k3,value3,recordjk3] = newton_A(x0);
[x4,val4,k4,value4,recordjk4] = Polak_Ribiere(x0)
[x5,val5,k5,value5,recordjk5] =  BFGS(x0);
[x6,val6,k6,value6,recordjk6] =  simplex(x0);
[v7,val7,flag7,k7,object_value7]=NMS(@sfun,[-0.75,1]);
x0 = [-0.75; 1];
historyx7 = object_value7(1,:);  
historyy7 = object_value7(2,:);
recordjk7 = log((historyx7 - 1).^2 + (historyy7 - 1).^2);
value7(:,1) = sfun(x0);
t = 0;
while t < k7 - 1
    t = t + 1;
    value7(:,t + 1) = sfun(object_value7(:,t + 1));
end
figure(1)
plot(recordjk1(1,:),'linewidth', 1.5)
hold on
plot(recordjk2(1,:),'linewidth', 1.5)
hold on
plot(recordjk3(1,:),'linewidth', 1.5)
hold on
plot(recordjk4(1,:),'linewidth', 1.5)
hold on
plot(recordjk5(1,:),'linewidth', 1.5)
hold on
plot(recordjk6(1,:),'linewidth', 1.5)
hold on
plot(recordjk7(1,:),'linewidth', 1.5)
grid on
xlabel('k')
ylabel('J')
legend('Gradient with Armijo','Newton without Armijo','Newton with Armijo','Polak-Ribiere with Armijo','BFGS with Armijo','Simplex','Modified Simplex')
xlim([1,inf])
title('Cost of Different Optimization Methods')
figure(2)
plot(value1(1,:),'linewidth', 1.5)
hold on
plot(value2(1,:),'linewidth', 1.5)
hold on
plot(value3(1,:),'linewidth', 1.5)
hold on
plot(value4(1,:),'linewidth', 1.5)
hold on
plot(value5(1,:),'linewidth', 1.5)
hold on
plot(value6(1,:),'linewidth', 1.5)
hold on
plot(value7(1,:),'linewidth', 1.5)
hold on
xlabel('k')
ylabel('v')
legend('Gradient with Armijo','Newton without Armijo','Newton with Armijo','Polak-Ribiere with Armijo','BFGS with Armijo','Simplex','Modified Simplex')
xlim([1,30])
ylim([0,15])
grid on
title('Value of Function versus k')


function [x,val,k,value,recordjk] = grad(x0)
%功能: 用最速下降法求解无约束问题:  min f(x)
%输入:  x0是初始点, fun, gfun分别是目标函数和梯度
%输出:  x, val分别是近似最优点和最优值,  k是迭代次数.
maxk = 5000;   %最大迭代次数
% rho = 0.5; sigma = 0.4;
alpha = 1; sigma = 0.5; gamma = 0.4;
k = 0;  epsilon = 1e-5;
history(:,1) = x0;
jk = log((x0(1)-1)^2 + (x0(2)-1)^2);
recordjk(:,1) = jk;
value(:,1) = fun(x0);
xk = x0;
while(k < maxk)
    g = gfun(xk);  %计算梯度
    dk = -g;    %计算搜索方向
    fprintf('DEBUG: %i %f\n',k,norm(g))
    if(norm(g) < epsilon), break; end
%     m = 0; mk = 0;
%     while(m < 100)   %Armijo搜索
%         if(fun(x0 + rho^m*d)) <(fun(x0) + sigma * rho^m * g' * d)
%             mk = m; break;
%         end
%         m = m+1;
%     end
%     x0 = x0 + rho^mk * d;
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
val = fun(xk); 
end


function [x,val,k,value,recordjk] = newton(x0)
%功能: 用牛顿法法求解无约束问题:  min f(x)
%输入:  x0是初始点, fun, gfun分别是目标函数和梯度
%输出:  x, val分别是近似最优点和最优值,  k是迭代次数.
d = 0.0;
k = 0;  epsilon = 1e-5;
history(:,1) = x0;
jk = log((x0(1)-1)^2 + (x0(2)-1)^2);
recordjk(:,1)=jk;
value(:,1) = fun(x0)
maxk = 5000;
xk = x0;
while (k < maxk) % first stop conditon
    [g,g2] = gfun2(xk);  %计算梯度
    d = -((g2))^-1 * g;
%     x0_1 = x0;
    if(norm(d) < epsilon), break; end
%     m = 0; mk = 0;
%     while(m < 100)   %Armijo搜索
%         if(fun(x0 + rho^m * d))< (fun(x0) + sigma * rho^m * g' * d)
%             mk = m; break;
%         end
%         m = m+1;
%     end
      fprintf('DEBUG: %i %f\n',k,norm(g))
%     x0 = x0 + rho^mk * d;
      xk = xk + d;

    k = k + 1;
    jk = log((xk(1)-1)^2 + (xk(2)-1)^2);
    recordjk(:,k + 1) = jk;
    history(:,k + 1) = xk;
    value(:,k + 1) = fun(xk)
end    
x = xk;
val = fun(xk); 
end

function [x,val,k,value,recordjk] = newton_A(x0)
%功能: 用牛顿法法求解无约束问题:  min f(x)
%输入:  x0是初始点, fun, gfun分别是目标函数和梯度
%输出:  x, val分别是近似最优点和最优值,  k是迭代次数.
d = 0.0;
k = 0; maxk = 5000; 
alpha = 2; sigma = 0.5; gamma = 0.4; epsilon = 1e-5;
history(:,1) = x0;
jk = log((x0(1)-1)^2 + (x0(2)-1)^2);
recordjk(:,1)=jk;
value(:,1) = fun(x0)
xk = x0;
while (k < maxk) % first stop conditon
    [g, g2] = gfun2(xk);  %计算梯度
    dk = -((g2))^-1 * g;
    fprintf('DEBUG: %i %f\n',k,norm(g))
    if(norm(dk) < epsilon)
        break; 
    end
    alphak = armijo (xk, alpha, sigma, gamma, dk);
    xk = xk + alphak * dk;
%     [g, g2] = gfun(xk);
%     if (abs(x0 - x0_1)< epsilon) %second stop condition
%         break;
%     end
    k = k + 1;
    jk = log((xk(1)-1)^2 + (xk(2)-1)^2);
    recordjk(:,k + 1) = jk;
    history(:,k + 1) = xk;
    value(:,k + 1) = fun(xk);
end    
x = xk;
val = fun(xk); 
end

function [x,val,k value,recordjk] = Polak_Ribiere(x0)
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
end

function [x,val,k,value,recordjk] = BFGS(x0)
%功能: 用BFGS求解无约束问题:  min f(x)
%输入:  x0是初始点, fun, gfun分别是目标函数和梯度
%输出:  x, val分别是近似最优点和最优值,  k是迭代次数.
maxk = 5000;   %最大迭代次数
H0=eye(2);
xk = x0; xk_1 = x0;
alpha = 1; sigma = 0.5; gamma = 0.4; epsilon = 1e-5;
k = 0;
history(:,1) = x0;
jk = log((x0(1)-1)^2 + (x0(2)-1)^2);
recordjk(:,1)=jk;
value(:,1) = fun(x0);
g = gfun(x0); dk = -g;
% r0 = g-gk_1;
% o0 = x0-xk_1;
% v0 = (r0'*H0*r0)^0.5 *(o0/(o0'*r0) - (H0*r0)/(r0'*H0*r0));
while k < maxk
    g = gfun(xk);  %计算梯度
    gk_1 = gfun(xk_1);
    fprintf('DEBUG: %i %f\n',k,norm(g))
    if(norm(g) < epsilon), break; end
    if k == 0
        Hk = H0;
    else
        rk=g-gk_1;
        ok=xk-xk_1;
        vk=(rk'*Hk*rk)^0.5 *(ok/(ok'*rk) - Hk*rk /(rk'*Hk*rk));
        Hk=Hk+ok*ok'/(ok'*rk) - Hk*rk*rk'*Hk/(rk'*Hk*rk)+vk*vk';
    end
    dk = -Hk*g;
    xk_1 = xk;
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
val = fun(xk); 
end

function [x,val,k,value,recordjk] = simplex(x0)
k = 0;maxk = 5000;
f0 = fun(x0);
stepsize = 1;
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
end

function [x,fval,flag,time,object_value]=NMS(fun,x0,max_time,eps)
% realization of Nelder-Mead Simplex
% max_time:the max number of iteration ,the default value is 10000
% eps: accuracy, the default value is 1e-8
% examine the parameters
if nargin < 2
    error('please enter 2 parameters at least, including funtion and starting point')
end
% set default value for parameter
if nargin < 3
    max_time = 10000;
end
if nargin < 4
    eps = 1e-5;
end
% Initialize  parameters
lambda=0.7; rho=1; chi=2; gama=0.5; sigma=0.5;
n=length(x0);% get dimension
p=[];
p(:,1)=x0(:); % transpose the matrix
object_value=x0(:);

% compulate the starting point
for i = 2:(n+1)
    e=zeros(n,1);
    e(i-1)=1;
    p(:,i)=p(:,1) + lambda * e;
end

% compulate the value
for i=1:(n+1)
    value (:,i)=feval(fun,p(:,i));
end

% sorting and ready to iterate
[value, index]=sort(value);
p=p(:,index);

color = [1 0 0]';
time = 1;

% start iteration
while max_time
    color = circshift(color, 2);
    fill3(p(1,:), p(2,:), [0 0 0], color')
    % breaking condition
%     if max(max(abs(p(:,2:n+1)-p(:,1:n)))) < eps
%         break
%     end
    mean=(value(1)+value(n)+value(n+1))/3;
    indicator=sqrt(((value(1)-mean)^2+(value(n)-mean)^2+(value(n+1)-mean)^2)/3);
    if indicator < eps
        break
    end
    % selet the three point
    best_point=p(:,1);
    best_point_value=value(:,1);
    sub_worst_point=p(:,n);
    sub_worst_point_value=value(:,n);
    worst_point=p(:,n+1);
    worst_point_value=value(:,n+1);
    
    % Reflection
    center=(best_point+sub_worst_point)/2;
    reflection_point=center+rho*(center-worst_point);
    reflection_point_value=feval(fun,reflection_point);
    
    if reflection_point_value < best_point_value
        % Expansion
        expansion_point=center+chi*(reflection_point-center);
        expansion_point_value=feval(fun,expansion_point);
        if expansion_point_value<reflection_point_value
            p(:,n+1)=expansion_point;
            value(:,n+1)=expansion_point_value;
        else
            p(:,n+1)=reflection_point;
            value(:,n+1)=reflection_point_value;
        end       
    else
        if reflection_point_value<sub_worst_point_value
            p(:,n+1)=reflection_point;
            value(:,n+1)=reflection_point_value;
        else
            % Outside Constraction
            shrink=0;
            if reflection_point_value < worst_point_value
                outside_constraction_point=center+gama*(reflection_point-center);
                outside_constraction_point_value=feval(fun,outside_constraction_point);
                if outside_constraction_point_value < worst_point_value
                    p(:,n+1)=outside_constraction_point;
                    value(:,n+1)=outside_constraction_point_value ;
                else
                    shrink=true;
                end
                % Inside Constraction
            else
                inside_constraction_point=center+gama*(worst_point-center);
                inside_constraction_point_value=feval(fun,inside_constraction_point);
                if inside_constraction_point_value < worst_point_value
                    p(:,n+1)=inside_constraction_point;
                    value(:,n+1)=inside_constraction_point_value ;
                else
                    shrink=1;
                end
            end
            
            % Shrinkage
            if shrink
                for i=2:n+1
                    p(:,i)=best_point+sigma*(p(:,i)-best_point);
                    value(:,i)=feval(fun,p(:,i));
                end
            end
        end
    end
    %  resort and ready to iterate
    [value, index] = sort(value);
    p = p(:,index);
    time = time + 1;
    object_value = [object_value p(:,1)];
end

x = p(:,1);
fval = value(:,1);
% time = max_time - time;
if max_time > 0
    flag = 1;
else
    flag = 0;
end

end


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

function [g,g2] = gfun2(x)
g = [400*x(1)*(x(1)^2-x(2))+2*(x(1)-1), -200*(x(1)^2-x(2))]';
g2 = [1200*x(1)^2-400*x(2)+2, -400*x(1); 
      -400*x(1), 200];
end

function f = sfun(v)
x = v(1);
y = v(2);
f = 100 * (x^2 - y)^2 + (x - 1)^2;
end

