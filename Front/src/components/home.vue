<template>
  <div>
    <v-container fluid pa-0>
      <v-row>
        <v-col class="pa-0">
          <v-parallax :src="getImgUrl" height="754"  align-center justify-center 
            data-aos="zoom-out"
            data-aos-duration="2000"
            data-aos-delay="50"
          >
            <v-row align="start" justify="center" no-gutters
        style="height: 1px; padding: 120px;" >
              <v-col class="text-center" cols="12" >
        <h1 style="color:#ffffff;
        font-size:100px; font-family:Arial;" 
        data-aos="fade-up"
        data-aos-duration="2000"
        data-aos-delay="50">Smart Editor</h1>
        <v-row>
        <v-col cols="3"><v-spacer/></v-col>
        <v-col cols="6"><div id="divider" style="background-color:white; height: 5px; width:100%;" 
        data-aos="zoom-in-right"
        data-aos-duration="1500"
        data-aos-delay="2000"></div></v-col>
            
        <v-col cols="3" lg="3" md="3" xs="2" sm="1"><v-spacer/></v-col>
        </v-row>
        <h2 class="subheading" id="test2"
        data-aos="fade-up"
        data-aos-duration="2000"
        data-aos-delay="2000">Change your Image</h2>
        <div class="m-5 p-5"
        data-aos="fade-up"
        data-aos-duration="2000"
        data-aos-delay="2000">
          <v-btn rounded x-large style="background-color:#663300;padding-left:90px;padding-right:90px;" dark @click="openDialog(true)"> 바로 시작하기 </v-btn>
        </div>
      </v-col>
              <v-col></v-col>
            </v-row>
          </v-parallax>
        </v-col>
      </v-row>
    </v-container>

    <section>
      <v-layout column wrap align-center>
        <v-flex xs12 sm4 class="my-4">
          <div class="text-center">
            <h3 data-aos="fade-down"
            data-aos-duration="1500"
            data-aos-delay="50" 
            style="color:#FF80AB; font-size:40px; font-weight:bold; font-family:noto sans kr; display: inline-block">드래그&드롭</h3>
            <h3 data-aos="fade-left"
            data-aos-duration="1500"
            data-aos-delay="50"
            style="font-size:35px; font-family:noto sans kr; font-weight:bold; display: inline-block">으로 누구나 쉽게</h3>
            <h2  class="headline" 
            data-aos="fade-left"
            data-aos-duration="1500"
            data-aos-delay="250"  style="font-size:30px; font-family:noto sans kr;">이미지를 편집할 수 있습니다.</h2>
            <br/>
            <p class="subheading"
            data-aos="fade-right"
            data-aos-duration="2000"
            data-aos-delay="350"
            style="font-family:noto sans kr; "
            ><strong>편집하는데</strong>귀찮고, <strong>시간이</strong>없는 분을 위해<br/>
            원하시는 <strong>사진</strong>을 올리시면, 당신의 <strong>사진을</strong>편집해 드립니다.</p>
          </div>
        </v-flex>

        <v-dialog v-model="dialog" persistent max-width="900px">
        <v-card>
          <v-card-title>
            <template>
              <v-icon style="margin-right:10px;" large color="#41B883" >cloud_upload</v-icon> 
              <span class="headline" large>파일 업로드</span>
            </template>
            <v-spacer></v-spacer>
            <v-btn icon @click="openDialog(false)"> <!-- closeDialog 클릭 이벤트 -->
              <v-icon>clear</v-icon>
            </v-btn>
          </v-card-title>
          <v-card-text>
            <v-row>
              <v-col cols="12" sm="12" md="12" style="position: relative; border:1px solid #41B883; border-style:dashed; ">
                <!-- 업로드 컴포넌트 -->
                <Upload :info="dialog" v-on:child="openDialog"></Upload> 


                <!-- <spinner :loading="loadingStatus"></spinner> -->

                
              </v-col>
            </v-row>
          </v-card-text>
        </v-card>
      </v-dialog>
        <!-- <v-flex xs12>
          <v-container grid-list-xl>
            <v-layout row wrap align-center>
              <v-flex xs12 md4>
                <v-card flat class="transparent">
                  <v-card-text class="text-center">
                    <v-icon x-large class="blue--text text--lighten-2">mdi-broom</v-icon>
                  </v-card-text>
                  <v-card-title primary-title class="layout justify-center">
                    <div class="headline text-center" 
                  data-aos="fade-right"
                  data-aos-duration="1500"
                  data-aos-delay="50">Service</div>
                  </v-card-title>
                  <v-card-text class="cardtext"
                  data-aos="fade-right"
                  data-aos-duration="2000"
                  data-aos-delay="50"
                  >
                    Lorem ipsum dolor sit amet, nemore sapientem ei qui,
                    no pri indoctum prodesset, omnis quidam utroque nam ei.
                    Ne utamur similique repudiandae eum.             
                  </v-card-text>
                </v-card>
              </v-flex>
              <v-flex xs12 md4>
                <v-card flat class="transparent">
                  <v-card-text class="text-center">
                    <v-icon x-large class="blue--text text--lighten-2">mdi-home-outline</v-icon>
                  </v-card-text>
                  <v-card-title primary-title class="layout justify-center">
                    <div class="headline"
                    data-aos="fade-down"
                    data-aos-duration="1500"
                    data-aos-delay="50"
                    >Service</div>
                  </v-card-title>
                  <v-card-text class="cardtext"
                  data-aos="fade-up"
                  data-aos-duration="2000"
                  data-aos-delay="50"
                  >
                    Lorem ipsum dolor sit amet, nemore sapientem ei qui,
                    no pri indoctum prodesset, omnis quidam utroque nam ei.
                    Ne utamur similique repudiandae eum.             
                  </v-card-text>
                </v-card>
              </v-flex>
              <v-flex xs12 md4>
                <v-card flat class="transparent">
                  <v-card-text class="text-center">
                    <v-icon x-large class="blue--text text--lighten-2">mdi-handshake</v-icon>
                  </v-card-text>
                  <v-card-title primary-title class="layout justify-center">
                    <div class="headline text-center"
                    data-aos="fade-left"
                    data-aos-duration="1500"
                    data-aos-delay="50"
                    >Service</div>
                  </v-card-title>
                  <v-card-text class="cardtext"
                  data-aos="fade-left"
                  data-aos-duration="2000"
                  data-aos-delay="50"
                  >
                    Lorem ipsum dolor sit amet, nemore sapientem ei qui,
                    no pri indoctum prodesset, omnis quidam utroque nam ei.
                    Ne utamur similique repudiandae eum.             
                  </v-card-text>
                </v-card>
              </v-flex>
            </v-layout>
          </v-container>
        </v-flex> -->
        
      </v-layout>
    </section>
      <v-row style="height: 150px; "></v-row>    
    <company></company>
    
  </div>
</template>


<script>

import company from './home/footer'
import { mapGetters } from 'vuex';
import Upload from "./home/ImageUploader.vue";

export default {
  
  name: 'home',
  components: {    
    company,
    Upload,
  },
  data(){
    return{
      dialog: false, //true : Dialog열림, false : Dialog닫힘
    };
  },
  methods: {
    openDialog(command) { //Dialog 열리는 동작
      if (!this.isAuthenticated) {
        alert("로그인 해주세요.");
      } else {
        if(command == true)
          this.dialog = true;
        else
          this.dialog = false
        }
    }
  },
  computed: {
    getImgUrl: function() {
      // return require("../assets/img/main.jpg");
      return require("../assets/back.jpg");
    },
    ...mapGetters([
      'isAuthenticated',
    ])
  },
}

</script>
