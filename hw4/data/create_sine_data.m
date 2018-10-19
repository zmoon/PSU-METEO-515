%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Generate propagating sine input data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear

ex = 5;
savename = ['sine_wave_data',num2str(ex),'.nc'];

% Creates Test Data
nx = 50;
T = 50;
x = 0:nx;
k = 2*pi/nx;
w = -2*pi/T;
time = 0:4*T;
nt = length(time);

if ex == 1 || ex == 2 || ex == 3
    A = ones(nt,1);
    k = 2*pi/nx*ones(nt,1);
elseif ex == 4
    A = 1+0.01*time;
    k = 2*pi/nx*ones(nt,1);
elseif ex == 5
    A = ones(nt,1);
    k = 2*pi/nx+0.01*sqrt(time);
end

c = 0;
for t = 1:nt
   c = c + 1;
   data(c,:) = A(t)*sin(k(t)*x+w*time(t)) ;
   if ex == 2 | ex == 3 | ex==4 | ex==5
       if time(t) == T || time(t) == 2*T || time(t) == 3*T || time(t) == 4*T ||  time(t) == T/2 || time(t) == T+T/2 ||time(t) == 2*T+(T/2) || time(t) == 3*T+(T/2) || time(t) == 4*T+(T/2)
          for d = 1:20
              c = c+1;
             data(c,:) = A(t)*sin(k(t)*x+w*time(t));

          end
       end
   end   
end

% Add Noise to Example 2
if ex == 3
   data = data + randn(size(data));
end


%% Save Data
% Save Data
[nt,nx] = size(data);
x = 1:length(x);
time = 1:length(nt);
nccreate(savename,'data','Dimensions',{'time',nt,'x',nx})
nccreate(savename,'x','Dimensions',{'x',nx})
nccreate(savename,'time','Dimensions',{'time',nt})
ncwrite(savename,'data',data)
ncwrite(savename,'x',x)
ncwrite(savename,'time',time);