function OD_est_value=OD(M,ts,D,lag,wi)

% ------------------------------------------------------------------------
% Input:
%          M:   number of multiple times series;
%          ts:  array of time series (time series can have different lengths L);
%          wl:  order or embedding dimension (it should be the same for all time series);
%          lag: vector of lags or embedding delay for the ordinal symbolization, e.g. [1,1,1] if M=3;
%          wi:  vector of weights for the generalized Jensen-Shannon divergence, e.g. [1/3,1/3,1/3] if M=3;
% Output:
%          OD_est_value: ordinal diversity estimated value
% Examples:
%          Example 1. OD estimated for M=100 normal white noises of length L=10^4 data points with order 
%          D=4, lags equal to 1 and weights equal to 1/100 for all time series:
%
%          OD_randn=OD(100,num2cell(randn(10000,100),1),4,ones(100,1)',1/100*ones(100,1)');
%
%          Example 2. OD estimated for M=9 simulations of fBms with Hurst exponents 0.1,0.2,...,0.9
%          (by using the MATLAB wfbm code) of length L=10^4 data points with order D=3, lags equal to 1 
%          and weights equal to 1/9 for all time series:
%
%          fBm=cell(9,1);
%          H=[0.1:0.1:0.9]';
%          for i=1:1:9;
%          fBm{i}=wfbm(H(i,1),10^4)';
%          end;
%          OD_fBm=OD(9,fBm,3,ones(9,1)',1/9*ones(9,1)');
%          clear fBm;
%          
% ------------------------------------------------------------------------
% Reference:
% If you use this algorithm please cite the following article:
% 
% "Quantifying the diversity of multiple time series with an ordinal symbolic approach" 
% by Luciano Zunino and Miguel C. Soriano, submitted to Phys. Rev. E, 2023
%
% Luciano Zunino
% E-mail: lucianoz@ciop.unlp.edu.ar, luciano.zunino@gmail.com
% ------------------------------------------------------------------------

p=zeros(factorial(D),M);
SPE=zeros(M,1);
for i=1:1:M
p(:,i)=prob_indices(ts{i},D,lag(i));
SPE(i,1)=SE(p(:,i));
end
OD_est_value=SE(sum(wi.*p,2))-wi*SPE;

function indcs=perm_indices(ts,D,lag)
m=length(ts)-(D-1)*lag;
indcs=zeros(m,1);
for i=1:D-1
st=ts(1+(i-1)*lag :m+(i-1)*lag);
for j=i:D-1
indcs=indcs+(st>ts(1+j*lag:m+j*lag));
end
indcs=indcs*(D-i);
end
indcs=indcs + 1;

function hist_indcs=hist_indices(ts,D,lag)
indcs=perm_indices(ts,D,lag);
hist_indcs=hist(indcs,1:1:factorial(D));

function prob_indcs=prob_indices(ts,D,lag)
m=length(ts)-(D-1)*lag;
prob_indcs=hist_indices(ts,D,lag)/m;

function y=SE(P)
P=P(P~=0);
y=-sum(P.*log(P));