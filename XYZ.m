%���ļ�������VLFEAT���Siftƥ��ģ��
clear;
close all;

B=-1000;%B<0,���ݻ����Ӿ���ppt����,�������ʵ(X,Z)ϵԭ��ΪOL���ģ�X����xL�᷽����ͬ��Z����ֱ����
f=1000;%f>0

%���ʵ�飨ͷ��
% for Gen=1:19
% clearvars -except B f Gen Rate RDG

%������ɲ��Լ���ѵ����
R(1,1:31)=randperm(41,31);
R(1,32:45)=41+randperm(19,14);
RT=1:60;
RT=setdiff(RT,R);

for t=1:2
    
    if t==2
        R=RT;
        %���Լ���ʱ��ʼ
        %         tic
    end
    
    for N=R
        clearvars -except B f N R2 Fv pv NIE R RT t RD Gen Rate RDG
        img=imread(['��׼������ͼƬ\',num2str(N),'.jpg']);%��ȡͼƬ�ļ���Ϣ������Ϊimg
        %ע��˫Ŀ�������ݿ�������ͼ��������������ͷ�ĵģ��Ұ����������ͷ�ĵġ���Ƭͼ������෴��
        [H,WB,~]=size(img);
        W=WB/2;
        
        imgLO=img(:,1:W,:);
        imgRO=img(:,(W+1):WB,:);
        
        imgR=im2single(rgb2gray(imgRO));%vl_sift��֧��single��������
        imgL=im2single(rgb2gray(imgLO));
        
        [FL,DL]=vl_sift(imgL);
        %F1�ڶ���Ϊ��������С������ȷ�������꣬��һ��Ϊ�����ꡣԭ����ͼƬ���Ͻǣ�y����ֱ���£�x��ˮƽ����
        [FR,DR]=vl_sift(imgR);
        
        %vl_feat�Դ�ƥ�䷽��
        ThresholdM=1.5;%Ĭ��1.5
        [M,L2D]=vl_ubcmatch(DL,DR,ThresholdM);
        %M�ĵ�һ��Ԫ��ΪFL����DL�����������ڶ���Ԫ��ΪFR����DR����������
        %M��ͬ�м�ƥ��㣬L2DΪ����DR��DR��L2Norm����
        
        xLO=FL(1,M(1,:));
        yLO=FL(2,M(1,:));
        xRO=FR(1,M(2,:));
        yRO=FR(2,M(2,:));
        
        WOT=W/2;
        xL=WOT-xLO;
        xR=WOT-xRO;
        yL=-yLO+H;
        yR=-yRO+H;
        
        %xL<xR�Ŷ�
        [~,NM]=size(M);
        Threshold_y=16;%ˮƽƥ��y��������
        j=1;
        Mark=zeros(1,NM);
        for i=1:NM
            if xL(i)-xR(i)>=0
                Mark(j)=i;
                j=j+1;
            else
                if abs(yL(i)-yR(i))>Threshold_y
                    Mark(j)=i;
                    j=j+1;
                end
            end
        end
        xL(Mark(Mark~=0))=[];
        xR(Mark(Mark~=0))=[];
        yL(Mark(Mark~=0))=[];
        yR(Mark(Mark~=0))=[];
        
        %����ɸѡЧ��
        xLO(Mark(Mark~=0))=[];
        xRO(Mark(Mark~=0))=[];
        yLO(Mark(Mark~=0))=[];
        yRO(Mark(Mark~=0))=[];
        xRO=xRO+size(imgL,2);
        
        %     % Siftƥ��Ч��ͼ
        %     figure;
        %     imshow([imgLO,imgRO]);
        %     hold on;
        %     h=line([xLO;xRO],[yLO;yRO]);
        
        X=B*xL./(xL-xR);%����ppt�Ƶ��Ĺ�ʽ
        Y=(yL+yR)/2;
        Z=B*f./(xL-xR);
        
        %     % ��ʾδ���ܶȾ���ĵ���
        %     figure;
        %     plot3(X,Y,Z,'^');
        %     xlabel('X');
        %     ylabel('Y');
        %     zlabel('Z');
        %     set(gca,'XDir','reverse');
        
        %�ܶȾ���DBSCAN��ȥ�������㲢��ȡǰ��
        Distance=zeros(length(X),length(X));
        Ep=200;
        MinPts=5;
        for i=1:length(X)
            for j=1:length(X)
                if i==j
                    Distance(i,j)=Ep+100;
                else
                    Distance(i,j)=((X(i)-X(j))^2+(Y(i)-Y(j))^2+(Z(i)-Z(j))^2)^0.5;
                end
            end
        end
        o=1;
        for i=1:length(X)
            Distance(i,j)=((X(i)-X(j))^2+(Y(i)-Y(j))^2+(Z(i)-Z(j))^2)^0.5;
            if sum(Distance(i,:)<=Ep)>=MinPts
                Omega(o)=i;
                o=o+1;
            end
        end
        k=0;
        Gamma=1:length(X);
        while isempty(Omega)==0
            GammaO=Gamma;
            r=ceil(length(Omega)*rand);
            l=1;
            Q(l)=Omega(r);
            Temp=intersect(Gamma,Q);
            Gamma=setdiff(Gamma,Temp);
            while isempty(Q)==0
                q=Q(1);
                Q(1)=[];
                if sum(Distance(q,:)<=Ep)>=MinPts
                    NEp=find(Distance(q,:)<=Ep);
                    delta=intersect(Gamma,NEp);
                    Q=[Q delta];
                    Temp=intersect(Gamma,delta);
                    Gamma=setdiff(Gamma,Temp);
                end
            end
            k=k+1;
            C{k}=setdiff(GammaO,intersect(GammaO,Gamma));
            Temp=intersect(Omega,C{k});
            Omega=setdiff(Omega,Temp);
        end
        
        %     % �����ܶȾ����ķ���ͼ
        %     ptsymb = {'bs','r^','md','go','c+','w*'};
        %     figure;
        %     for i=1:k
        %         plot3(X(C{i}),Y(C{i}),Z(C{i}),ptsymb{i});
        %         hold on
        %     end
        %     xlabel('X');
        %     ylabel('Y');
        %     zlabel('Z');
        %     grid on
        
        %��һ�����˺��Ż�ǰ������ȡ��������Ƭ��ӦԪ��
        index=C{1};
        Threshold_FB=500;%����ǰ���ھ���������Ȳ�
        if k~=1
            for i=1:k
                MCZ(i)=mean(Z(C{i}));
                NC(i)=length(Z(C{i}));
            end
            [~,Mi]=min(MCZ);
            [~,MMi]=max(MCZ);
            [~,Mn]=max(NC);
            for i=2:k
                index=[index C{i}];
            end
            %�����Զ���Ĵ�Ԫ����Ŀ�������ģ���ɾ���ô�
            if Mi~=Mn
                index=setdiff(index,C{Mi});
            end
            %����ǳ�������Ĵ���Ԫ����Ŀ���Ĵص���СԪ����Ȳ����ֵ�����ʹ����ǳ�Ĵؽ��з���
            if abs(min(Z(C{MMi}))-max(Z(C{Mn})))>Threshold_FB
                index=C{MMi};
            end
        end
        
        %���Իع�
        [BB,~,~,~,stats]=regress((Z(index))',[ones(length(index),1) (X(index))' (Y(index))']);
        R2(N)=stats(1);%R��
        Fv(N)=stats(2);%Fֵ
        pv(N)=stats(3);%pֵ
        NIE(N)=length(index);
    end
    
    %SVM�����ŷָ�ֵ
    % if t==1
    % RD=mean([max(R2(R(R<=41))) min(R2(R(R>41)))]);
    % end
    
    %���Լ���ʱ����
    % if t==2
    %     toc
    % end
    
end

%�����жϣ����ȫ��ͼƬ��
k=1;
Threshold_Flesh_R2=RD;%R���Ļ�������ֵ
for N=RT
    if R2(N)<Threshold_Flesh_R2
        Output(k,:)=[N 1];
        k=k+1;
    else
        Output(k,:)=[N 0];
        k=k+1;
    end
end
Answer=zeros(15,1);
for i=1:15
    if Output(i,1)<=41
        Answer(i)=1;
    end
end

%���ʵ�飨β��
% Rate(Gen)=sum(Output(:,2)==Answer)/15*100;%���Լ���ȷ��
% RDG(Gen)=RD;%�ָ��
%
% end

% % ���ƻع�ƽ�棨��Ե�һͼƬ��
% xfit = min(X(index)):5:max(X(index));%ע 0.1��ʾ���ݵļ��
% yfit = min(Y(index)):5:max(Y(index));
% [XFIT,YFIT]= meshgrid (xfit,yfit);%�Ƴ���������
% ZFIT = BB(1) + BB(2) * XFIT + BB(3) * YFIT;
% mesh (XFIT,YFIT,ZFIT);

% ����R��ͼ�����ȫ��ͼƬ��
% figure
% plot([1:15],R2(R2~=0),'*');
% hold on;
% plot([1:15],R2(R2~=0));
% title('R��');

% % ����Fֵͼ�����ȫ��ͼƬ��
% figure
% plot([1:45],Fv,'*');
% hold on;
% plot([1:45],Fv);
% title('Fֵ');

% % ������Ч������ͼ�����ȫ��ͼƬ��
% figure;
% plot([1:60],NIE,'*');
% hold on;
% plot([1:60],NIE);
