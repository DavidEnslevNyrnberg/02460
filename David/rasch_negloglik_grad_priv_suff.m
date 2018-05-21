function [negloglik, grad ] = rasch_negloglik_grad_priv_suff(w, xd_priv,xn_priv,lam)
%  Detailed explanation goes here
% computes the normalized neg log-likelihood of the Rasch model
% based on sufficients statistics sum(X,1) and sum(X,2)
% lam is L2 regularization
    D=size(xn_priv,2);
    N=size(xd_priv,1);
    bet=w(1:N);
    del=w((N+1):(N+D))';
    dum=exp(bet*ones(1,D)+ones(N,1)*del);
    dum1=1./(1 + dum );
    negloglik=sum(sum(log(dum1)))-sum(xd_priv.*bet)-sum(xn_priv.*del)+lam*(sum(bet.*bet)+sum(del.*del));
    pp=dum.*dum1;
    ped=sum(pp,2);
    pen=sum(pp,1);
    gradbet=(ped-xd_priv)+2*lam*bet;
    graddel=(pen-xn_priv)+2*lam*del;
    grad=[gradbet;graddel'];
end

