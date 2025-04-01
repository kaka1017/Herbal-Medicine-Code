function [P,XS_sub,XT_sub,u_sc,u_subc] = CSBD(XS,label_XS,XT,CDD,dim,lamda,gamma)

%Calculate the center of c-class samples from source domain in the original space
u_sc = cell(5,1);
for i = 1:5
    if size(find(label_XS == i),1) ~= 0
        XS_i = XS(:,find(label_XS == i));
        u_sc{i} = sum(XS_i,2)/size(XS_i,2);
    end
end

% Calculate the center of the source domain and target domain in the original space
us = sum(XS,2)/size(XS,2);
ut = sum(XT,2)/size(XT,2);

% Calculate marginal distribution distance in the original space
MDD = (us-ut)*(us-ut)';

% Calculate distribution divergence in the original space
if CDD == 0
    Distance = MDD;%Only MDD is calculated when the first iteration
else
    Distance = (1-gamma)*MDD + gamma*CDD;%
end

%Calculate the energy of the source domain and target domain in the original space
E =  (1-lamda)*(XS*XS') + lamda*(XT*XT');

%Calculate the optimal basis transformation P
A = pinv(E) * Distance;
[P,D] = eig(A);
[~,Idex] = sort(diag(D));
P = real(P(:,Idex(1:dim)));

%Calculate source domain and target domain in the subspace
XS_sub = P' * XS;
XT_sub = P' * XT;

%Calculate the center of c-class samples from source domain in the subspace
u_subc = cell(6,1);
for i = 1:6
    if ~isempty(find(label_XS == i,1))
        k = find(label_XS == i);
        XS_subi = XS_sub(:,k);
        u_subc{i} = sum(XS_subi,2)/size(XS_subi,2);
    end
end