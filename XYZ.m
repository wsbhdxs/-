%本文件调用了VLFEAT库的Sift匹配模块
clear;
close all;

B=-1000;%B<0,根据机器视觉课ppt定义,物体的真实(X,Z)系原点为OL光心，X轴与xL轴方向相同，Z轴竖直向上
f=1000;%f>0

%多次实验（头）
% for Gen=1:19
% clearvars -except B f Gen Rate RDG

%随机生成测试集和训练集
R(1,1:31)=randperm(41,31);
R(1,32:45)=41+randperm(19,14);
RT=1:60;
RT=setdiff(RT,R);

for t=1:2
    
    if t==2
        R=RT;
        %测试集计时开始
        %         tic
    end
    
    for N=R
        clearvars -except B f N R2 Fv pv NIE R RT t RD Gen Rate RDG
        img=imread(['标准化测试图片\',num2str(N),'.jpg']);%读取图片文件信息并命名为img
        %注意双目人脸数据库中真人图像左半边是右摄像头拍的，右半边是左摄像头拍的。照片图像与此相反。
        [H,WB,~]=size(img);
        W=WB/2;
        
        imgLO=img(:,1:W,:);
        imgRO=img(:,(W+1):WB,:);
        
        imgR=im2single(rgb2gray(imgRO));%vl_sift仅支持single数据类型
        imgL=im2single(rgb2gray(imgLO));
        
        [FL,DL]=vl_sift(imgL);
        %F1第二行为行数，即小数级精确的纵坐标，第一行为横坐标。原点在图片左上角，y轴竖直向下，x轴水平向右
        [FR,DR]=vl_sift(imgR);
        
        %vl_feat自带匹配方案
        ThresholdM=1.5;%默认1.5
        [M,L2D]=vl_ubcmatch(DL,DR,ThresholdM);
        %M的第一行元素为FL（即DL）的列数，第二行元素为FR（即DR）的列数。
        %M的同列即匹配点，L2D为向量DR和DR的L2Norm距离
        
        xLO=FL(1,M(1,:));
        yLO=FL(2,M(1,:));
        xRO=FR(1,M(2,:));
        yRO=FR(2,M(2,:));
        
        WOT=W/2;
        xL=WOT-xLO;
        xR=WOT-xRO;
        yL=-yLO+H;
        yR=-yRO+H;
        
        %xL<xR才对
        [~,NM]=size(M);
        Threshold_y=16;%水平匹配y允许的误差
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
        
        %测试筛选效果
        xLO(Mark(Mark~=0))=[];
        xRO(Mark(Mark~=0))=[];
        yLO(Mark(Mark~=0))=[];
        yRO(Mark(Mark~=0))=[];
        xRO=xRO+size(imgL,2);
        
        %     % Sift匹配效果图
        %     figure;
        %     imshow([imgLO,imgRO]);
        %     hold on;
        %     h=line([xLO;xRO],[yLO;yRO]);
        
        X=B*xL./(xL-xR);%根据ppt推导的公式
        Y=(yL+yR)/2;
        Z=B*f./(xL-xR);
        
        %     % 显示未经密度聚类的点云
        %     figure;
        %     plot3(X,Y,Z,'^');
        %     xlabel('X');
        %     ylabel('Y');
        %     zlabel('Z');
        %     set(gca,'XDir','reverse');
        
        %密度聚类DBSCAN，去除大误差点并提取前景
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
        
        %     % 绘制密度聚类后的分类图
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
        
        %进一步过滤和优化前景，提取人脸或照片对应元素
        index=C{1};
        Threshold_FB=500;%允许前景内景物的最大深度差
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
            %若最深（远）的簇元素数目不是最多的，则删除该簇
            if Mi~=Mn
                index=setdiff(index,C{Mi});
            end
            %若最浅（近）的簇与元素数目最多的簇的最小元素深度差超过阈值，则仅使用最浅的簇进行分析
            if abs(min(Z(C{MMi}))-max(Z(C{Mn})))>Threshold_FB
                index=C{MMi};
            end
        end
        
        %线性回归
        [BB,~,~,~,stats]=regress((Z(index))',[ones(length(index),1) (X(index))' (Y(index))']);
        R2(N)=stats(1);%R方
        Fv(N)=stats(2);%F值
        pv(N)=stats(3);%p值
        NIE(N)=length(index);
    end
    
    %SVM求最优分割值
    % if t==1
    % RD=mean([max(R2(R(R<=41))) min(R2(R(R>41)))]);
    % end
    
    %测试集计时结束
    % if t==2
    %     toc
    % end
    
end

%活体判断（针对全部图片）
k=1;
Threshold_Flesh_R2=RD;%R方的活体检测阈值
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

%多次实验（尾）
% Rate(Gen)=sum(Output(:,2)==Answer)/15*100;%测试集正确率
% RDG(Gen)=RD;%分割点
%
% end

% % 绘制回归平面（针对单一图片）
% xfit = min(X(index)):5:max(X(index));%注 0.1表示数据的间隔
% yfit = min(Y(index)):5:max(Y(index));
% [XFIT,YFIT]= meshgrid (xfit,yfit);%制成网格数据
% ZFIT = BB(1) + BB(2) * XFIT + BB(3) * YFIT;
% mesh (XFIT,YFIT,ZFIT);

% 绘制R方图（针对全部图片）
% figure
% plot([1:15],R2(R2~=0),'*');
% hold on;
% plot([1:15],R2(R2~=0));
% title('R方');

% % 绘制F值图（针对全部图片）
% figure
% plot([1:45],Fv,'*');
% hold on;
% plot([1:45],Fv);
% title('F值');

% % 绘制有效点数量图（针对全部图片）
% figure;
% plot([1:60],NIE,'*');
% hold on;
% plot([1:60],NIE);
