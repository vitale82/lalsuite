REAL8 tmp1=coeffs->KK*eta;
REAL8 tmp3=-1.+tmp1;
REAL8 tmp7=(1.0/(x->data[0]*x->data[0]));
REAL8 tmp10=1/x->data[0];
REAL8 tmp6=sigmaKerr->data[2]*sigmaKerr->data[2];
REAL8 tmp14=x->data[0]*x->data[0];
REAL8 tmp4=(1.0/(tmp3*tmp3));
REAL8 tmp5=1.*tmp4;
REAL8 tmp8=1.*tmp6*tmp7;
REAL8 tmp9=1/tmp3;
REAL8 tmp11=2.*tmp10*tmp9;
REAL8 tmp12=tmp11+tmp5+tmp8;
REAL8 tmp17=coeffs->k0*eta;
REAL8 tmp18=1./(x->data[0]*x->data[0]*x->data[0]*x->data[0]);
REAL8 tmp19=1.*coeffs->k4*tmp18;
REAL8 tmp20=1./(x->data[0]*x->data[0]*x->data[0]);
REAL8 tmp21=1.*coeffs->k3*tmp20;
REAL8 tmp22=1.*coeffs->k2*tmp7;
REAL8 tmp23=1.*coeffs->k1*tmp10;
REAL8 tmp24=pow(x->data[0],-5.);
REAL8 tmp25=1.*tmp10;
REAL8 tmp26=log(tmp25);
REAL8 tmp27=coeffs->k5l*tmp26;
REAL8 tmp28=coeffs->k5+tmp27;
REAL8 tmp29=1.*tmp24*tmp28;
REAL8 tmp30=1.+tmp19+tmp21+tmp22+tmp23+tmp29;
REAL8 tmp31=log(tmp30);
REAL8 tmp32=eta*tmp31;
REAL8 tmp33=1.+tmp17+tmp32;
REAL8 tmp15=0.+tmp14;
REAL8 tmp16=1/tmp15;
REAL8 tmp41=-3.*eta;
REAL8 tmp42=4.+tmp41;
REAL8 tmp43=1.*p->data[0];
REAL8 tmp44=0.+tmp43;
REAL8 tmp47=0.+p->data[0];
REAL8 tmp48=1.*tmp47;
REAL8 tmp49=0.+tmp48;
REAL8 tmp50=26.+tmp41;
REAL8 tmp51=2.*eta*tmp20*tmp50;
REAL8 tmp52=6.*eta*tmp7;
REAL8 tmp53=1.+tmp51+tmp52;
REAL8 tmp54=log(tmp53);
REAL8 tmp55=1.+tmp54;
REAL8 tmp60=1.*e3z;
REAL8 tmp61=0.+tmp60;
REAL8 tmp35=tmp14+tmp6;
REAL8 tmp36=tmp35*tmp35;
REAL8 tmp37=-(tmp12*tmp14*tmp33*tmp6);
REAL8 tmp38=tmp36+tmp37;
REAL8 tmp80=sqrt(tmp6);
REAL8 tmp75=1/tmp38;
REAL8 tmp84=(x->data[0]*x->data[0]*x->data[0]);
REAL8 tmp90=eta*eta;
REAL8 tmp94=(tmp49*tmp49*tmp49);
REAL8 tmp95=tmp12*tmp12;
REAL8 tmp96=pow(x->data[0],6.);
REAL8 tmp97=(1.0/(tmp15*tmp15));
REAL8 tmp98=tmp55*tmp55;
REAL8 tmp99=tmp33*tmp33;
REAL8 tmp62=1.*tmp61;
REAL8 tmp63=0.+tmp62;
REAL8 tmp64=0.+p->data[2];
REAL8 tmp65=tmp63*tmp64;
REAL8 tmp66=0.+tmp65;
REAL8 tmp67=tmp66*tmp66;
REAL8 tmp68=1.*tmp14*tmp16*tmp67;
REAL8 tmp69=tmp49*tmp49;
REAL8 tmp70=1.*tmp12*tmp14*tmp16*tmp33*tmp55*tmp69;
REAL8 tmp71=0.+p->data[1];
REAL8 tmp72=tmp61*tmp71;
REAL8 tmp73=0.+tmp72;
REAL8 tmp74=tmp73*tmp73;
REAL8 tmp76=1.*tmp14*tmp15*tmp74*tmp75;
REAL8 tmp101=-16.*eta;
REAL8 tmp102=21.*tmp90;
REAL8 tmp103=tmp101+tmp102;
REAL8 tmp108=0.+tmp68+tmp70+tmp76;
REAL8 tmp107=(x->data[0]*x->data[0]*x->data[0]*x->data[0]);
REAL8 tmp122=-6.*eta;
REAL8 tmp123=39.*tmp90;
REAL8 tmp124=tmp122+tmp123;
REAL8 tmp136=sqrt(tmp15);
REAL8 tmp85=-66.*sigmaKerr->data[2]*tmp12*tmp16*tmp33*tmp49*tmp55*tmp84;
REAL8 tmp86=-52.*sigmaStar->data[2]*tmp12*tmp16*tmp33*tmp49*tmp55*tmp84;
REAL8 tmp87=tmp85+tmp86;
REAL8 tmp88=0.08333333333333333*eta*tmp10*tmp87;
REAL8 tmp89=103.*eta;
REAL8 tmp91=-60.*tmp90;
REAL8 tmp92=tmp89+tmp91;
REAL8 tmp93=-4.*tmp12*tmp16*tmp33*tmp49*tmp55*tmp84*tmp92;
REAL8 tmp100=1440.*tmp90*tmp94*tmp95*tmp96*tmp97*tmp98*tmp99;
REAL8 tmp104=-12.*tmp103*tmp94*tmp95*tmp96*tmp97*tmp98*tmp99;
REAL8 tmp105=3.*eta;
REAL8 tmp106=23.+tmp105;
REAL8 tmp109=-4.*eta*tmp106*tmp107*tmp108*tmp12*tmp16*tmp33*tmp49*tmp55;
REAL8 tmp110=-47.*eta;
REAL8 tmp111=54.*tmp90;
REAL8 tmp112=tmp103*tmp108*x->data[0];
REAL8 tmp113=tmp110+tmp111+tmp112;
REAL8 tmp114=-12.*tmp113*tmp12*tmp16*tmp33*tmp49*tmp55*tmp84;
REAL8 tmp115=tmp100+tmp104+tmp109+tmp114+tmp93;
REAL8 tmp116=0.013888888888888888*sigmaStar->data[2]*tmp115*tmp7;
REAL8 tmp117=-109.*eta;
REAL8 tmp118=51.*tmp90;
REAL8 tmp119=tmp117+tmp118;
REAL8 tmp120=8.*tmp119*tmp12*tmp16*tmp33*tmp49*tmp55*tmp84;
REAL8 tmp121=3240.*tmp90*tmp94*tmp95*tmp96*tmp97*tmp98*tmp99;
REAL8 tmp125=-12.*tmp124*tmp94*tmp95*tmp96*tmp97*tmp98*tmp99;
REAL8 tmp126=-180.*eta*tmp107*tmp108*tmp12*tmp16*tmp33*tmp49*tmp55;
REAL8 tmp127=16.*eta;
REAL8 tmp128=147.*tmp90;
REAL8 tmp129=tmp108*tmp124*x->data[0];
REAL8 tmp130=tmp127+tmp128+tmp129;
REAL8 tmp131=-12.*tmp12*tmp130*tmp16*tmp33*tmp49*tmp55*tmp84;
REAL8 tmp132=tmp120+tmp121+tmp125+tmp126+tmp131;
REAL8 tmp133=0.006944444444444444*sigmaKerr->data[2]*tmp132*tmp7;
REAL8 tmp134=tmp116+tmp133+tmp88;
REAL8 tmp159=(tmp49*tmp49*tmp49*tmp49);
REAL8 tmp162=tmp108*tmp108;
REAL8 tmp137=tmp12*tmp14*tmp33;
REAL8 tmp138=sqrt(tmp137);
REAL8 tmp139=-tmp138;
REAL8 tmp140=tmp12*tmp14*tmp15*tmp33*tmp75;
REAL8 tmp141=sqrt(tmp140);
REAL8 tmp142=1.*tmp136*tmp141;
REAL8 tmp143=tmp139+tmp142;
REAL8 tmp144=1.+tmp68+tmp70+tmp76;
REAL8 tmp147=1.*coeffs->d1v2*eta*sigmaKerr->data[2]*tmp20;
REAL8 tmp148=-8.*sigmaKerr->data[2];
REAL8 tmp149=14.*sigmaStar->data[2];
REAL8 tmp150=-36.*sigmaKerr->data[2]*tmp12*tmp16*tmp33*tmp55*tmp69*tmp84;
REAL8 tmp151=-30.*sigmaStar->data[2]*tmp12*tmp16*tmp33*tmp55*tmp69*tmp84;
REAL8 tmp152=3.*sigmaKerr->data[2]*tmp108*x->data[0];
REAL8 tmp153=4.*sigmaStar->data[2]*tmp108*x->data[0];
REAL8 tmp154=tmp148+tmp149+tmp150+tmp151+tmp152+tmp153;
REAL8 tmp155=0.08333333333333333*eta*tmp10*tmp154;
REAL8 tmp156=27.*eta;
REAL8 tmp157=-353.+tmp156;
REAL8 tmp158=-2.*eta*tmp157;
REAL8 tmp160=360.*tmp159*tmp90*tmp95*tmp96*tmp97*tmp98*tmp99;
REAL8 tmp161=-2.*tmp108*tmp92*x->data[0];
REAL8 tmp163=-(eta*tmp106*tmp14*tmp162);
REAL8 tmp164=-6.*tmp113*tmp12*tmp16*tmp33*tmp55*tmp69*tmp84;
REAL8 tmp165=tmp158+tmp160+tmp161+tmp163+tmp164;
REAL8 tmp166=0.013888888888888888*sigmaStar->data[2]*tmp165*tmp7;
REAL8 tmp167=8.+tmp105;
REAL8 tmp168=-112.*eta*tmp167;
REAL8 tmp169=810.*tmp159*tmp90*tmp95*tmp96*tmp97*tmp98*tmp99;
REAL8 tmp170=4.*tmp108*tmp119*x->data[0];
REAL8 tmp171=-45.*eta*tmp14*tmp162;
REAL8 tmp172=-6.*tmp12*tmp130*tmp16*tmp33*tmp55*tmp69*tmp84;
REAL8 tmp173=tmp168+tmp169+tmp170+tmp171+tmp172;
REAL8 tmp174=0.006944444444444444*sigmaKerr->data[2]*tmp173*tmp7;
REAL8 tmp175=0.+sigmaStar->data[2]+tmp147+tmp155+tmp166+tmp174;
REAL8 tmp13=1/tmp12;
REAL8 tmp34=1/tmp33;
REAL8 tmp182=tmp12*tmp14*tmp33*tmp55;
REAL8 tmp183=sqrt(tmp182);
REAL8 tmp184=sqrt(tmp144);
REAL8 tmp177=1./sqrt(tmp15);
REAL8 tmp187=tmp107*tmp55*tmp95*tmp99;
REAL8 tmp188=1./sqrt(tmp187);
REAL8 tmp189=-2.*tmp12*tmp14*tmp33;
REAL8 tmp190=1/tmp30;
REAL8 tmp191=2.*coeffs->k2;
REAL8 tmp192=3.*coeffs->k3;
REAL8 tmp193=4.*coeffs->k4;
REAL8 tmp194=5.*tmp10*tmp28;
REAL8 tmp195=tmp193+tmp194;
REAL8 tmp196=1.*tmp10*tmp195;
REAL8 tmp197=tmp192+tmp196;
REAL8 tmp198=1.*tmp10*tmp197;
REAL8 tmp199=tmp191+tmp198;
REAL8 tmp200=1.*tmp10*tmp199;
REAL8 tmp201=coeffs->k1+tmp200;
REAL8 tmp202=-(eta*tmp12*tmp190*tmp201);
REAL8 tmp203=1.*tmp9;
REAL8 tmp204=1.*tmp10*tmp6;
REAL8 tmp205=tmp203+tmp204;
REAL8 tmp206=-2.*tmp205*tmp33;
REAL8 tmp207=2.*tmp12*tmp33*x->data[0];
REAL8 tmp208=tmp202+tmp206+tmp207;
REAL8 tmp209=tmp183*tmp208;
REAL8 tmp210=tmp189+tmp209;
REAL8 tmp145=1./sqrt(tmp144);
REAL8 tmp216=1.*tmp16*x->data[0];
REAL8 tmp217=-4.*tmp12*tmp33*tmp84;
REAL8 tmp218=tmp208*tmp35;
REAL8 tmp219=tmp217+tmp218;
REAL8 tmp220=0.5*tmp13*tmp219*tmp34*tmp35*tmp7*tmp75;
REAL8 tmp221=tmp216+tmp220;
REAL8 tmp213=tmp175*tmp63;
REAL8 tmp214=0.+tmp213;
REAL8 tmp81=0.;
REAL8 tmp82=2.*tmp80*x->data[0];
REAL8 tmp83=tmp81+tmp82;
REAL8 tmp211=1.+tmp184;
REAL8 tmp185=1.+tmp184+tmp68+tmp70+tmp76;
REAL8 tmp56=2.*tmp12*tmp14*tmp16*tmp33*tmp49*tmp55;
REAL8 tmp231=1./sqrt(tmp137);
REAL8 tmp232=1./sqrt(tmp140);
REAL8 tmp233=(1.0/(tmp38*tmp38));
REAL8 tmp234=2.*tmp80;
REAL8 tmp235=0.;
REAL8 tmp236=0.+tmp234+tmp235;
REAL8 tmp237=tmp236*tmp38;
REAL8 tmp238=-4.*tmp35*x->data[0];
REAL8 tmp239=1.*tmp208*tmp6;
REAL8 tmp240=tmp238+tmp239;
REAL8 tmp241=tmp240*tmp83;
REAL8 tmp242=tmp237+tmp241;
REAL8 tmp244=tmp15*tmp15;
REAL8 tmp256=pow(tmp15,-2.5);
REAL8 tmp178=(1.0/sqrt(tmp144*tmp144*tmp144));
REAL8 tmp243=1/tmp211;
REAL8 tmp259=1.*tmp107*tmp12*tmp214*tmp244*tmp33*tmp74*tmp75;
REAL8 tmp260=-(tmp12*tmp14*tmp214*tmp33*tmp55*tmp69);
REAL8 tmp261=tmp15*tmp185*tmp214;
REAL8 tmp262=0.+tmp260+tmp261;
REAL8 tmp263=1.*tmp12*tmp14*tmp262*tmp33;
REAL8 tmp264=0.+tmp259+tmp263;
REAL8 tmp249=1.*tmp12*tmp14*tmp145*tmp16*tmp33*tmp49*tmp55;
REAL8 tmp250=tmp249+tmp56;
REAL8 tmp222=2.*tmp184;
REAL8 tmp223=1.+tmp222;
REAL8 tmp39=tmp13*tmp16*tmp34*tmp38*tmp7;
REAL8 tmp40=1./sqrt(tmp39);
REAL8 tmp58=(tmp44*tmp44*tmp44*tmp44);
REAL8 tmp59=2.*eta*tmp42*tmp58*tmp7;
REAL8 tmp77=1.+tmp59+tmp68+tmp70+tmp76;
REAL8 tmp179=e3z*tmp175;
REAL8 tmp180=0.+tmp179;
REAL8 tmp230=(1.0/sqrt(tmp15*tmp15*tmp15));
REAL8 tmp186=1/tmp185;
REAL8 tmp268=-0.5*tmp136*tmp141*tmp188*tmp210*tmp211*tmp214*tmp73*x->data[0];
REAL8 tmp269=1.*tmp136*tmp141*tmp214*tmp221*tmp223*tmp73*x->data[0];
REAL8 tmp270=0.+tmp269;
REAL8 tmp271=tmp138*tmp270;
REAL8 tmp272=tmp268+tmp271;
REAL8 tmp273=tmp183*tmp272;
REAL8 tmp274=0.+tmp273;
REAL8 dHdp0=(1.*eta*(-(tmp134*tmp175*tmp20)-0.5*tmp12*tmp14*tmp178*tmp183*tmp231*tmp232*tmp233*tmp242*tmp243*tmp256*tmp264*tmp33*tmp49*tmp55+0.5*tmp145*tmp183*tmp230*tmp231*tmp232*tmp233*tmp242*tmp243*(1.*tmp12*tmp14*tmp33*(tmp15*tmp214*tmp250-2.*tmp12*tmp14*tmp214*tmp33*tmp49*tmp55+tmp134*tmp15*tmp185*tmp63-tmp12*tmp134*tmp14*tmp33*tmp55*tmp63*tmp69)+1.*tmp107*tmp12*tmp134*tmp244*tmp33*tmp63*tmp74*tmp75)+1.*e3z*tmp134*tmp75*tmp83-tmp12*tmp143*tmp177*tmp178*tmp180*tmp33*tmp49*tmp55*tmp73*tmp75*tmp84+1.*e3z*tmp134*tmp136*tmp143*tmp145*tmp73*tmp75*x->data[0]+1.*tmp13*tmp141*tmp16*tmp183*tmp186*tmp34*tmp7*(-0.5*tmp12*tmp141*tmp145*tmp177*tmp188*tmp210*tmp214*tmp33*tmp49*tmp55*tmp73*tmp84-0.5*tmp134*tmp136*tmp141*tmp188*tmp210*tmp211*tmp63*tmp73*x->data[0]+tmp138*(2.*tmp12*tmp141*tmp145*tmp177*tmp214*tmp221*tmp33*tmp49*tmp55*tmp73*tmp84+1.*tmp134*tmp136*tmp141*tmp221*tmp223*tmp63*tmp73*x->data[0]))-tmp13*tmp141*tmp16*tmp250*tmp274*tmp34*tmp7*(1.0/(tmp185*tmp185))-(0.5*tmp12*tmp14*tmp183*tmp231*tmp232*tmp233*tmp242*tmp256*tmp264*tmp33*tmp49*tmp55*(1.0/(tmp211*tmp211)))/tmp144+(0.5*tmp40*(tmp56+8.*eta*tmp42*tmp7*(tmp44*tmp44*tmp44)))/sqrt(tmp77)))/sqrt(1.+2.*eta*(-1.+0.5*tmp145*tmp183*tmp230*tmp231*tmp232*tmp233*tmp242*tmp243*tmp264+1.*tmp13*tmp141*tmp16*tmp186*tmp274*tmp34*tmp7+1.*tmp180*tmp75*tmp83+1.*tmp136*tmp143*tmp145*tmp180*tmp73*tmp75*x->data[0]+1.*tmp73*tmp75*tmp83*x->data[0]+1.*coeffs->dheffSSv2*eta*tmp18*(s1Vec->data[2]*s1Vec->data[2]+s2Vec->data[2]*s2Vec->data[2])-0.5*tmp20*(0.+tmp175*tmp175)+tmp40*sqrt(tmp77)));
