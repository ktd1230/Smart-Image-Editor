<template>
  <v-app-bar style="color: white; background: url('https://cdn.pixabay.com/photo/2018/12/20/16/24/star-3886297__340.jpg')">
      <v-container
        mx-auto
        pa-0
      >
        <v-layout class="align-center">
          <v-img
            :src="require('@/assets/logo.png')"
            class="mr-5"
            contain
            height="55"
            width="55"
            max-width="55"
            @click="switchPage('home')"
          />
          
          <!-- <v-btn
            v-for="(link, i) in links"
            :key="i"
            :to="link.to"
            class="ml-0 hidden-sm-and-down"
            flat
            @click="onClick($event, item)"
          >
            {{ link.text }}
          </v-btn> -->
          <!-- <v-btn text @click="switchPage('home')">Home</v-btn> -->
          <!-- <v-btn style="color:white;" text v-if="isAuthenticated" @click="switchPage('mybook')">Mybook</v-btn> -->
          <!-- <v-btn style="color:white;" text @click="switchPage('result')">result</v-btn> -->          
          <!-- <v-btn  class="ml-0 hidden-sm-and-down" text>test</v-btn> -->
          <!-- <v-btn  class="ml-0 hidden-sm-and-down" text>test</v-btn> -->
          <v-spacer />
          
          <div v-if="isAuthenticated">
            <span>{{username}}님 환영합니다. </span>
            <v-btn class="ml-4" color="blue darken-1" text @click="logout()">로그아웃</v-btn>
          </div>

          <div v-else class="text-center">
            <v-menu
              v-model="signupMenu"
              :close-on-content-click="false"
              :nudge-width="200"
              :close-delay="5000"
              :open-on-hover="false"
              offset-y
            >
              <template v-slot:activator="{ on }" >
                <v-btn style="color:white;" v-on="on" text class="mr-5" 
              @click="errors = []; credential.password=''; credential.username=''; password2='';">
                  회원가입
                </v-btn>
              </template>
              <v-card>
                <v-card-title>
                  <span class="headline">회원가입</span>
                </v-card-title>
                <v-card-text>
                  <v-container>
                    <v-row>
                      <v-col cols="12">
                        <v-text-field label="ID" v-model="credential.username" required style="padding:0px;margin:0px;"></v-text-field>
                        <v-text-field label="Password" v-model="password2" type="password" required style="padding:0px;margin:0px;"></v-text-field>
                        <v-text-field label="Password2" v-model="credential.password" type="password" required style="padding:0px;margin:0px;" @keydown.enter="signup()"></v-text-field>
                      </v-col>
                    </v-row>
                    <div v-if="errors.length">
                      <div v-for="(error, idx) in errors" :key="idx" class="mx-auto text-danger">
                          {{error}}
                      </div>
                    </div>
                  </v-container>
                </v-card-text>
                <v-card-actions>
                  <v-spacer></v-spacer>
                  <v-btn color="blue darken-1" text @click="signupMenu = false;">취소</v-btn>
                  <v-btn color="blue darken-1" text @click="signup();">확인</v-btn>
                </v-card-actions>
              </v-card>
            </v-menu>
          
          
            <v-menu
              v-model="loginMenu"
              :close-on-content-click="false"
              :nudge-width="200"
              :close-delay="5000"
              :open-on-hover="false"
              offset-y
            >
              <template v-slot:activator="{ on }">
                <v-btn color="primary" v-on="on"
              @click="errors = []; credential2.username=''; credential2.password='';">
                  <span>로그인</span>
                </v-btn>
              </template>
              <v-card>
                <v-card-title>
                  <span class="headline">로그인</span>
                </v-card-title>
                <v-card-text>
                  <v-container>
                    <v-row>
                      <v-col cols="12">
                        <v-text-field label="ID" v-model="credential2.username" required style="padding:0px;margin:0px;"></v-text-field>
                        <v-text-field label="Password" v-model="credential2.password" type="password" required class="p-0 m-0" style="padding:0px;margin:0px;" @keydown.enter="login()"></v-text-field>
                      </v-col>
                    </v-row>
                    <div v-if="errors.length">
                      <div v-for="(error, idx) in errors" :key="idx" class="mx-auto text-danger">
                          {{error}}
                      </div>
                    </div>
                  </v-container>
                </v-card-text>
                <v-card-actions>
                  <v-spacer></v-spacer>
                  <v-btn color="blue darken-1" text @click="loginMenu = false;">취소</v-btn>
                  <v-btn color="blue darken-1" text @click="login();">확인</v-btn>
                </v-card-actions>
              </v-card>
            </v-menu>
          </div>
          
          <v-btn v-if="isAuthenticated" icon @click="openDialog(true)"><v-icon color="white">mdi-plus</v-icon></v-btn>
        </v-layout>
      </v-container>


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
    </v-app-bar>
</template>

<script>
import Upload from "./ImageUploader.vue";
import axios_common from '../../axios_common';
import { mapGetters } from 'vuex';

export default {
  components: {
    Upload,
  },
  computed: {
    ...mapGetters([
      'isAuthenticated',
      'requestHeader',
      'userId',
      'username'
    ])
  },
  data: ()=> ({
    dialog: false, //true : Dialog열림, false : Dialog닫힘
    signupMenu: false,
    loginMenu: false,
    // loadingStatus: false,
    password2: '',
    credential: {
      username: '',
      password: ''
    },
    credential2: {
      username: '',
      password: ''
    },
    errors: [],
  }),
  methods: {
    openDialog(command) { //Dialog 열리는 동작
    if(command == true)
      this.dialog = true;
    else
      this.dialog = false
    },
    signup(){
        console.log('회원가입 시도')
        console.log(this.credential)
        const form = new FormData()
        form.append('username', this.credential.username)
        form.append('password', this.credential.password)
        console.log(form)
        if (this.signupCheckForm()){
          axios_common.post('/accounts/signup/', form)
            .then((res)=>{
                console.log(res)
                console.log('회원가입 성공')
                this.signupMenu = false;
            })
            .catch((e)=>{
                console.log(e)
                this.errors.push('회원가입 실패')
            })
        }
    },
    signupCheckForm(){
        this.errors = []
        if (this.password2.length < 8 || this.credential.password.length < 8) {this.errors.push('비밀번호는 8글자가 넘어야합니다.')}
        if (!this.credential.username) {this.errors.push('아이디를 입력해주세요.')}
        if (this.password2 != this.credential.password){
          this.errors.push('비밀번호가 일치하지 않습니다.')
        } else {
            if(!this.errors.length){return true}
        }
    },

    login(){
      if (this.checkForm()){
        console.log('로그인 시도')
        axios_common.post('/api-token-auth/', this.credential2)
          .then((res)=>{
            console.log('로그인 성공')

            this.$store.dispatch('login', res.data.token)
            this.loginMenu = false;
          })
          .catch((e)=>{
            console.log(e)
            this.errors.push('로그인 실패')
          })
        }
    },
    checkForm(){
        // 배열초기화로 데이터가 쌓이는 것 방지
        this.errors = []
        if (this.credential2.password.length < 8) {this.errors.push('비밀번호는 8글자가 넘어야합니다.')}
        if (!this.credential2.username) {this.errors.push('아이디를 입력해주세요.')}
        if (this.errors.length === 0) {return true}
    },
    
    logout(){
      this.$store.dispatch('logout')
    },
    switchPage(page) {
      if (page == "home") {
        page = "";
      }
      this.$router.push(`/${page}`);
    },
  }
}
</script>

<style>
.btn { 
  display:block; width:200px; height:40px; line-height:40px; 
  border:1px #3399dd solid;; margin:15px auto; 
  background-color:#66aaff; text-align:center; 
  cursor: pointer; color:#333; transition:all 0.9s, color 0.3; 
  } 
  
.btn:hover{color:#fff;}

.hovery:hover{ 
  box-shadow: 200px 0 0 0 rgba(0,0,0,0.25) inset, 
  -200px 0 0 0 rgba(0,0,0,0.25) inset; 
  }

</style>
