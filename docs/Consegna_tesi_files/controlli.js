var obj;
var XMLHTTP;
var OggettoXMLHTTP;
var oprel;

//**********************************************
//Funzioni utilizzate da tesi0.asp
//**********************************************

function verificaaut(){
	document.getElementById("prosegui").disabled = !document.getElementById("flagletturadomanda").checked;
	if (document.getElementById("flagletturadomanda").checked==false){
		document.getElementById("prosegui").disabled=true;
	} else {
		document.getElementById("prosegui").disabled=false;
	}
}

function controllainformativa(){
	w=window.document.frm_tesi;
	if (w.flag_lettura_domanda.checked==true){
		w.flag_lettura_domanda.disabled=false;
		w.method = "POST";
		w.action = "esegui_tesi0.asp";
		w.submit();
	}
}	

//**********************************************
//Funzioni utilizzate da tesi1.asp
//**********************************************

function copiarel(txt,val){
	window.document.frm_tesi.mat_relatore.options[document.frm_tesi.mat_relatore.options.length] = new Option(txt, val);
}


function aggiungirel(matrelatore,relatore){
var url = "eseguirelatore.asp";
w=window.document.frm_tesi;
oprel='ins';
XMLHTTP = RilevaBrowser(CambioStato);
XMLHTTP.open("POST", url, true);
XMLHTTP.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
XMLHTTP.send("op=i&matrelatore="+matrelatore+"&relatore="+relatore.replace(/ /g,'%20'));
}
function CambioStato(){
    if (XMLHTTP.readyState == 4 && oprel == 'ins'){
        window.close();
    }
}
function RilevaBrowser(TipoBrowser){
	if (navigator.userAgent.indexOf("MSIE") != (-1)){
		var Classe = "Msxml2.XMLHTTP";
		if (navigator.appVersion.indexOf("MSIE 5.5") != (-1));{
			Classe = "Microsoft.XMLHTTP";
		} 
		try{
			OggettoXMLHTTP = new ActiveXObject(Classe);
			OggettoXMLHTTP.onreadystatechange = TipoBrowser;
			return OggettoXMLHTTP;
		}
		catch(e){alert("Errore: gen.3");}
	} else {
	OggettoXMLHTTP = new XMLHttpRequest();
	OggettoXMLHTTP.onload = TipoBrowser;
	OggettoXMLHTTP.onerror = TipoBrowser;
	return OggettoXMLHTTP;
	}
}

function rimuovirel(){
	w=window.document.frm_tesi;
	del=0;
	q=w.mat_relatore.length;
		for (x=0;x<q;x++){
			if (w.mat_relatore.options[x].selected==true){
				matrelatore=w.mat_relatore.options[x].value;
				relatore=w.mat_relatore.options[x].text;
				w.mat_relatore.options[x]=null;
				del=1;
				break;
			}
		}

	if (del==1){
		var url = "eseguirelatore.asp";
		w=window.document.frm_tesi;
		oprel='del';
		XMLHTTP = RilevaBrowser(CambioStato);
		XMLHTTP.open("POST", url, true);
		XMLHTTP.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
		XMLHTTP.send("op=d&matrelatore="+matrelatore+"&relatore="+relatore);
	} else {
		alert ('Selezionare il relatore da eliminare!');
		w.mat_relatore.focus();
		return;
	}
}

function aggiungiling(){
	w=window.document.frm_tesi;
	xq=w.cod_lingua_tesi_tab.length;
	for (x=0;x<xq;x++){
		if (w.cod_lingua_tesi_tab.options[x].selected==true){
			valore=w.cod_lingua_tesi_tab.options[x].value;
			testo=w.cod_lingua_tesi_tab.options[x].text;
			w.cod_lingua_tesi.options[w.cod_lingua_tesi.length]=new Option(testo, valore, false, false);
		}
	}

	//la rimozione delle voci selezionate deve essere eseguita in un ciclo for separato
	yq=w.cod_lingua_tesi.length;
	for (y=0;y<yq;y++){
	
		for (x=0;x<xq;x++){
			if (w.cod_lingua_tesi.options[y].value==w.cod_lingua_tesi_tab.options[x].value){
				w.cod_lingua_tesi_tab.options[x]=null;
				break;
			}
		}
		
	}
}

function rimuoviling(){
	w=window.document.frm_tesi;
	xq=w.cod_lingua_tesi.length;
	for (x=0;x<xq;x++){
		if (w.cod_lingua_tesi.options[x].selected==true){
			valore=w.cod_lingua_tesi.options[x].value;
			testo=w.cod_lingua_tesi.options[x].text;
			w.cod_lingua_tesi_tab.options[w.cod_lingua_tesi_tab.length]=new Option(testo, valore, false, false);
		}
	}
	
	//la rimozione delle voci selezionate deve essere eseguita in un ciclo for separato
	yq=w.cod_lingua_tesi_tab.length;
	for (y=0;y<yq;y++){
	
		for (x=0;x<xq;x++){
			if (w.cod_lingua_tesi_tab.options[y].value==w.cod_lingua_tesi.options[x].value){
				w.cod_lingua_tesi.options[x]=null;
				break;
			}
		}
	}
}

function controllaDim(campo,lunghmax,contatore) {
	var lunghtotale = campo.value.length; 
	if(lunghtotale >= lunghmax) {
		campo.value = campo.value.substring(0, lunghmax);
	}
	document.getElementById(contatore).innerHTML = lunghmax-campo.value.length;
}

/*function verificagb(){
	w=window.document.frm_tesi;
	if (document.getElementById("flagsupplement").checked==true){
		document.getElementById('obbgb').style.display='';
	} else {
		document.getElementById('obbgb').style.display='none';
	}	
}*/

function prosegui_upload1(){
	w=window.document.frm_tesi;

	Filtro = /^([0-9\,])+$/;
	var re=/\s+$|^\s+/g;

	w.ind_resid.value=w.ind_resid.value.replace(re,"");	
	w.numtel_resid.value=w.numtel_resid.value.replace(re,"");
	w.auth_email.value=w.auth_email.value.replace(re,"");	
	w.cattedra.value=w.cattedra.value.replace(re,"");	
	w.cod_cattedra.value=w.cod_cattedra.value.replace(re,"");	
	w.mat_relatore.value=w.mat_relatore.value.replace(re,"");	
	w.cod_lingua_tesi.value=w.cod_lingua_tesi.value.replace(re,"");	
	w.titolo.value=w.titolo.value.replace(re,"");	
	w.titolo_inglese.value=w.titolo_inglese.value.replace(re,"");
	//w.flag_supplement.disabled=false;

	if (w.dst.value!='tesi0'){
		if (w.ind_resid.value==''){
			alert ('Attenzione inserire l\'indirizzo di residenza');
			w.ind_resid.focus();
			return;
		}
	
		if (Filtro.test(w.numtel_resid.value)==false && w.numtel_resid.value!=''){
			alert ('Attenzione inserire un valore numerico nel riferimento telefonico');
			w.numtel_resid.focus();
			return;	
		}
	
		if (w.auth_email.value!='SI' && w.auth_email.value!='NO'){
			alert ('Attenzione indicare se l\'email deve essere resa pubblica');
			w.auth_email.focus();
			return;
		}
	
		if (w.cattedra.value==''){
			alert ('Attenzione inserire l\'insegnamento relativo alla tesi');
			w.cattedra.focus();
			return;
		}
	
		if (w.cod_cattedra.value==''){
			alert ('Attenzione inserire il codice insegnamento');
			w.cod_cattedra.focus();
			return;
		}
	
		if (w.mat_relatore.length==0){
			alert ('Attenzione inserire uno o piu\' relatori');
			w.mat_relatore.focus();
			return;
		} else {
			q=w.mat_relatore.length;
			for (x=0;x<q;x++){
				w.mat_relatore.options[x].selected=true;
			}	
		}
	
		if (w.cod_lingua_tesi.length==0){
			alert ('Attenzione selezionare la lingua di redazione della tesi');
			w.cod_lingua_tesi_tab.focus();
			return;
		} else {
			q=w.cod_lingua_tesi.length;
			for (x=0;x<q;x++){
				w.cod_lingua_tesi.options[x].selected=true;
			}	
		}
	
		if (w.titolo.value==''){
			alert ('Attenzione inserire il titolo della tesi');
			w.titolo.focus();
			return;
		}
		
		/*if (w.flag_supplement.checked==true && w.titolo_inglese.value==''){
			alert ('Attenzione il diploma Supplement richiede obbligatoriamente l\'inserimento del titolo tesi in inglese');
			q=w.mat_relatore.length;
			for (x=0;x<q;x++){
				w.mat_relatore.options[x].selected=false;
			}	
			q=w.cod_lingua_tesi.length;
			for (x=0;x<q;x++){
				w.cod_lingua_tesi.options[x].selected=false;
			}	
			w.titolo_inglese.focus();
			return;
		}*/
	} else {
		if (w.cod_lingua_tesi.length!=0){
			q=w.cod_lingua_tesi.length;
			for (x=0;x<q;x++){
				w.cod_lingua_tesi.options[x].selected=true;
			}	
		}
	}
	
	w.titolo.disabled=false;
	w.titolo_inglese.disabled=false;
	w.titolo_altra_lingua.disabled=false;

	document.getElementById('attesa').style.display='';
	document.getElementById('salvaprogress').style.display='';
	w.method="post";
	w.action="esegui_tesi1.asp";
	w.submit();
}

//**********************************************
//Funzioni utilizzate da tesi2.asp
//**********************************************

function formaAnnoAcc(valore,modulo,annoattuale){
	if (modulo.anno_accademico_1.value.length < 4){modulo.anno_accademico_2.value=''}
	if (modulo.anno_accademico_1.value.length==4){
		if (valore>annoattuale || valore<1960){
			alert("ATTENZIONE: il campo Anno accademico deve contenere un valore compreso fra 1960 e "+annoattuale+"!");
			modulo.anno_accademico_1.value='';
			modulo.anno_accademico_2.value='';
			modulo.anno_accademico_1.focus();
			return false;
		}
		document.getElementById("anno_accademico_2").value=(Number(valore)+1).toString().substr(2);
	}
}

function prosegui_upload2(destinazione){
	w=window.document.frm_tesi;

	Filtro = /^([0-9\,])+$/;
	var re=/\s+$|^\s+/g;

	w.id_area.value=w.id_area.value.replace(re,"");	
	w.id_tipologia.value=w.id_tipologia.value.replace(re,"");
	w.descrizione_breve.value=w.descrizione_breve.value.replace(re,"");	
	w.anno_accademico_1.value=w.anno_accademico_1.value.replace(re,"");	
	w.anno_accademico_2.value=w.anno_accademico_2.value.replace(re,"");	
	w.pc1.value=w.pc1.value.replace(re,"");	
	w.pc2.value=w.pc2.value.replace(re,"");	
	w.pc3.value=w.pc3.value.replace(re,"");	
	w.pc4.value=w.pc4.value.replace(re,"");	
	w.pc5.value=w.pc5.value.replace(re,"");	
	w.pc6.value=w.pc6.value.replace(re,"");	
	w.pc7.value=w.pc7.value.replace(re,"");	
	w.pc8.value=w.pc8.value.replace(re,"");	
	w.pc9.value=w.pc9.value.replace(re,"");	
	w.pc10.value=w.pc10.value.replace(re,"");	
	w.pag.value=destinazione;

	if (w.pag.value=='next'){
		if (w.id_area.value==''){
			alert ('Attenzione selezionare l\'area disciplinare');
			w.id_area.focus();
			return;
		}
	
	
		if (w.id_tipologia.value==''){
			alert ('Attenzione selezionare la tipologia di tesi');
			w.id_tipologia.focus();
			return;
		}
	
		if (w.descrizione_breve.value==''){
			alert ('Attenzione inserire l\'abstract');
			w.descrizione_breve.focus();
			return;
		}
		
		if (Filtro.test(w.anno_accademico_1.value)==false){
			alert ('Attenzione inserire un valore numerico nell\'anno accademico');
			w.anno_accademico_1.focus();
			return;	
		}

		if (w.anno_accademico_1.value=='' || w.anno_accademico_2.value==''){
			alert ('Attenzione inserire l\'anno accademico');
			w.anno_accademico_1.focus();
			return;
		}
	
		pchiave=0;
		if (w.pc1.value!=''){pchiave=pchiave+1;}
		if (w.pc2.value!=''){pchiave=pchiave+1;}
		if (w.pc3.value!=''){pchiave=pchiave+1;}
		if (w.pc4.value!=''){pchiave=pchiave+1;}
		if (w.pc5.value!=''){pchiave=pchiave+1;}
		if (w.pc6.value!=''){pchiave=pchiave+1;}
		if (w.pc7.value!=''){pchiave=pchiave+1;}
		if (w.pc8.value!=''){pchiave=pchiave+1;}
		if (w.pc9.value!=''){pchiave=pchiave+1;}
		if (w.pc10.value!=''){pchiave=pchiave+1;}
	
		if (pchiave < 2){
			alert ('Attenzione inserire almeno due parole chiave');
			w.pc1.focus();
			return;
		}
	} else {
		if (w.st.value=='4' && Filtro.test(w.anno_accademico_1.value)==false){
			alert ('Attenzione inserire un valore numerico nell\'anno accademico');
			w.anno_accademico_1.focus();
			return;	
		}
		if (w.st.value=='4' && (w.anno_accademico_1.value=='' || w.anno_accademico_2.value=='')){
			alert ('Attenzione inserire l\'anno accademico');
			w.anno_accademico_1.focus();
			return;
		}
	}

	document.getElementById('attesa').style.display='';
	document.getElementById('salvaprogress').style.display='';
	w.method="post";
	w.action="esegui_tesi2.asp";
	w.submit();
}

//**********************************************
//Funzioni utilizzate da tesi3.asp
//**********************************************

function prosegui_upload3(modulo){
	var strPath;
	if(modulo.elements["blob"].value == ""){
		alert("ATTENZIONE: selezionare un file da inviare!");
        modulo.elements["blob"].focus();
		return false;
	}
	strPath=document.getElementById("blobId").value;
	strPath=strPath.toString().replace("/","\\");
	if(strPath.indexOf("\\")>-1){
		strPath=strPath.substr(strPath.lastIndexOf("\\")+1);
	 }
	document.getElementById('attesa').style.display='';
	document.getElementById('salvaprogress').style.display='';
	strPath=replaceNomeFile(strPath);
	modulo.action="esegui_tesi3.asp?f="+strPath+"&anno_accademico="+modulo.anno_accademico.value;
	modulo.submit();
}

function replaceNomeFile(strPath){
	var re=/[^a-zA-Z-_.^1-90 ]|\^/g; /*ammessi solo lettere senza accenti, numeri e simboli . - _*/
	strPath=strPath.toString().replace(re,"");
	return strPath;
}

//**********************************************
//Funzioni utilizzate da tesi4.asp
//**********************************************

function prosegui_upload4(modulo){
	var strPath;
	if(modulo.elements["blob"].value == ""){
		alert("ATTENZIONE: selezionare un file da inviare!");
        modulo.elements["blob"].focus();
		return false;
	}
	strPath=document.getElementById("blobId").value;
	strPath=strPath.toString().replace("/","\\");
	if(strPath.indexOf("\\")>-1){
		strPath=strPath.substr(strPath.lastIndexOf("\\")+1);
	}
	document.getElementById('attesa').style.display='';
	document.getElementById('salvaprogress').style.display='';
	strPath=replaceNomeFile(strPath);
	modulo.action="esegui_tesi4.asp?f="+strPath+"&anno_accademico="+modulo.anno_accademico.value;
	modulo.submit();
}

//******************************************************************
//Funzioni utilizzate da menu.asp/tesionline.asp x gestione box tesi
//******************************************************************
function check_link(val){
	if(val=='N'){
		document.getElementById('link_upd').style.display='';
		document.getElementById('link_info').style.display='none';
	} else {
		document.getElementById('link_upd').style.display='none';
		document.getElementById('link_info').style.display='';
		}
}

function set_upd(){
	w=window.document.frm_tesi;
	/*valorizza gli hidden di menu.asp*/
	var str=w.tol_idcorso.value;
	var fac_cds=str.split('@-@');
	w.tol_esse3_stu_id.value=fac_cds[0];
	w.tol_esse3_fac_id.value=fac_cds[1];
	w.tol_esse3_cds_id.value=fac_cds[2];
	w.tol_id_upload.value='';
	w.method="post";
	w.action="tesi/set_tesi0.asp";
	w.submit();
}

function set_mod_upd(stuid_tmp,facid_tmp,cdsid_tmp){
	w=window.document.frm_tesi;
	w.tol_esse3_stu_id.value=stuid_tmp;
	w.tol_esse3_fac_id.value=facid_tmp;
	w.tol_esse3_cds_id.value=cdsid_tmp;
	w.method="post";
	w.action="tesi/set_tesi0.asp";
	w.submit();
}

function vis_upd(idupload_tmp){
	w=window.document.frm_tesi;
	w.tol_id_upload.value=idupload_tmp;
	w.method="post";
	w.action="tesi/set_tesi5.asp";
	w.submit();
}
