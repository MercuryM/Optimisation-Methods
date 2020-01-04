clear
clc
close all
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
%------------------------------------A7M----------------------------------%
%[v_1,fval_1]=fminsearch(@fun,[0.6,0.6])
%[v_2,fval_2]=fminsearch(@fun,[-0.8,1])
% select different starting point
[v_1,val_1,flag_1,time_1,object_value_1]=NMS(@fun,[-0.75,1]);
% [v_2,val_2,flag_2,time_2,object_value_2]=NMS(@fun,[-0.8,1]);
% [v_3,val_3,flag_3,time_3,object_value_3]=NMS(@fun,[0.5,1.7]);
% selet some of iteration data
x0 = [-0.75; 1];
historyx = object_value_1(1,:);  
historyy = object_value_1(2,:);
recordjk = log((historyx - 1).^2 + (historyy-1).^2);
value(:,1) = fun(x0);
t = 0;
while t < time_1 - 1
    t = t + 1;
    value(:,t + 1) = fun(object_value_1(:,t + 1));
end
% value = feval(fun,object_value_1);
% x_2 = object_value_2(1,1:20);  
% y_2 = object_value_2(2,1:20);
% x_3 = object_value_3(1,1:20);  
% y_3 = object_value_3(2,1:20);
% plot(x_1,y_1,'-or',x_2,y_2,'-*b',x_3,y_3,'-dk');
plot(historyx,historyy,'r*-');
text(historyx(1),historyy(1),['Initial point'],'color',[1 0.5 0])
text(historyx(t + 1),historyy(t + 1),['Terminate point'],'color',[1 0.5 0])
% some customization of this image
title('Modified Simplex Method');
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

function f = fun(v)
x = v(1);
y = v(2);
f = 100 * (x^2 - y)^2 + (x - 1)^2;
end










% 
% f = @(x) 100 * (x(2) - x(1)^2)^2 + (1 - x(1))^2;
% %x0 = [3 - 6 * rand(), 0.5 - rand()]'
% x0 = [-3/4 1]'
% 
% virtualization = true;
% 
% if virtualization
% 	[X,Y] = meshgrid(-2.2:0.2:2.2, -2.8:0.2:6);
% 	Z = zeros(size(X));
% 	for i = 1:numel(X)
% 		Z(i) = f([X(i) Y(i)]');
% 	end
% 	hold on
% % 	grid on
% 
% 	mesh(X, Y, Z)
% 	alpha(0.3)
% 	contour(X, Y, Z)
% 	color = [1 0 0]';
% end
% 
% init_step = 1;
% maxsteps = 99;
% cntstep = 0;
% good_delta = 0.00001;
% dim = length(x0);
% 
% % initial simlex points
% x = zeros(dim, dim + 1);
% scores = zeros(1, dim + 1);
% x(:, 1) = x0;
% for i = 1 : dim
% 	x(:, i + 1) = x0;
% 	x(i, i + 1) = x(i, i + 1) + init_step;
% end
% 
% % main loop
% while cntstep < maxsteps
% 	if virtualization
% 		color = circshift(color, 2);
% 		fill3(x(1,:), x(2,:), [0 0 0], color')
% 	end
% 
% 	% update scores
% 	for i = 1 : dim + 1
% 		scores(i) = f(x(:, i));
% 	end
% 
% 	% sorting simplex vertex by scores
% 	[scores, idx] = sort(scores);
% 	x = x(:, idx);
% 	scores;
% 
% 	% termination condition	
% 	if cntstep > 0
% 		delta = abs(scores(dim + 1) - prev_worst_score);
% 		if delta < good_delta
% 			cntstep
% 			display('Good enough.')
% 			break
% 		end
% 	end
% 	prev_worst_score = scores(dim + 1);
% 
% 	% calculate m and r (reflected point)
% 	m = sum(x(:,1:dim)')' ./ dim;
% 	r = 2*m - x(:, dim + 1);
% 
% 	if f(x(:,1)) <= f(r) && f(r) < f(x(:,dim))
% 		x(:, dim + 1) = r;
% 		display('Reflect (1)')
% 
% 	elseif f(r) < f(x(:,1))
% 		% calculate expansion point s
% 		s = m + 2*(m - x(:, dim + 1));
% 		if f(s) < f(r)
% 			x(:, dim + 1) = s;
% 			display('Expand')
% 		else
% 			x(:, dim + 1) = r;
% 			display('Reflect (2)')
% 		end
% 	else
% 		do_shrink = false;
% 		% perform a contraction between m and the better of x(n+1) and r
% 		if f(r) < f(x(:, dim + 1))
% 			% better than the wrost one
% 			c = m + (r - m) ./ 2;
% 			if f(c) < f(r)
% 				x(:, dim + 1) = c;
% 				display('Contract outside')
% 			else
% 				% otherwise, continue with Shrink step
% 				do_shrink = true;
% 			end
% 		else
% 			% even wrose than the wrost one
% 			cc = m + (x(:, dim + 1) - m) ./ 2;
% 			if f(cc) < f(x(:, dim + 1))
% 				x(:, dim + 1) = cc;
% 				display('Contract inside')
% 			else
% 				% otherwise, continue with Shrink step
% 				do_shrink = true;
% 			end
% 		end
% 
% 		if do_shrink
% 			% Shrink
% 			for i = 1 : dim
% 				x(:, i + 1) = x(:, 1) + (x(:, i + 1) - x(:, 1)) ./ 2;
% 			end
% 			display('Shrink')
% 		end
% 	end
% 	cntstep = cntstep + 1;
% 	input('please hit enter ...');
% end
% 
% ret_x = x(:, 1)
% ret_score = scores(1)
