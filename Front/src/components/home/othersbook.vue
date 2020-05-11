<template>
    <v-container fluid class="pa-0" style="background: #f4f2f3;">

        <v-row style="height: 130px;"></v-row>
          <v-row >
            <!-- <v-col cols="12" class="pa-0">
              <v-sheet
          class="d-flex"
          color="transparent"
          height="100"
        >
        </v-sheet>
        
            </v-col> -->
            <v-col cols="1">
            <v-spacer/>
        </v-col>
                        <v-col lg="3" md="4" xs="4" sm="4">
                             <p data-aos="fade-left"
            data-aos-duration="1500"
            data-aos-delay="350" 
            class="mb-0"
            style="font-size:40px; font-family:noto sans kr;">화제의</p>
            <h3 data-aos="fade-left"
            data-aos-duration="1500"
            data-aos-delay="550" 
            style="font-size:40px; font-weight:bold; font-family:noto sans kr; display: inline-block">이야기들</h3>
                        </v-col>
                        <v-col lg="4" md="4" xs="5" sm="5" align-self="end">
                             <p
                    data-aos="fade-left"
                    data-aos-duration="3000"
                    data-aos-delay="1550"
                    class="mb-0"
            style="font-size:16px; font-family:noto sans kr;"
                    >알파북이 어떤 이야기를 만들까요?<br/>

                        미리 작성된 이야기를 확인해 보세요.
                    </p>
                          
                   
                        </v-col>
                         

                    </v-row>
        <v-row style="height: 100px;"></v-row>
     

        <v-row >
            <v-carousel
                hide-delimiters="hide-delimiters"
                style="box-shadow: 0px 0px"
                height="300px">
                <v-carousel-item v-for="i in 2" :key="i">
                    <v-row>
                        <v-col md="4" xs="12" sm="12" v-for="(story, jj) in storys" :key="jj" pl-2="pl-2" pr-2="pr-2">
                            <!-- {{i}} -->
                            <!-- {{jj}} -->
                            <v-hover>
                                <template v-slot:default="{ hover }">
                                    <v-card height="400px" aspect-ratio="0.7">
                                        <v-img :src="`http://i02c104.p.ssafy.io:8000/media/${storys[jj+((i-1)*3)].firstimage}`" height="200px"  
                                        data-aos="fade-up"
                                        data-aos-delay="100"
                                        data-aos-once="true"
                                        data-aos-duration="3000"
                                        ></v-img>
                                        <v-card-title primary-title="primary-title">
                                            <div>
                                                <h3 class="headline mb-0">제목 : {{storys[jj+((i-1)*3)].title}}</h3>
                                                <p>작성자 : {{storys[jj+((i-1)*3)].user.username}}</p>
                                            </div>
                                        </v-card-title>
                                        <v-fade-transition>
                                            <v-overlay v-if="hover" absolute="absolute" color="#036358">
                                                <v-btn @click="detail(storys[jj+((i-1)*3)])">자세히 보기</v-btn>
                                            </v-overlay>
                                        </v-fade-transition>
                                    </v-card>
                                </template>
                            </v-hover>
                        </v-col>
                    </v-row>
                </v-carousel-item>
            </v-carousel>
        </v-row>

        <v-row style="height: 100px;"></v-row>

        <section style="background: white;padding-top:80px;">
            <v-row>
                <v-col cols="7">
                    <v-layout column wrap class="my-8" align-center>
                        <v-flex xs12 sm4 class="my-4">
                        <div class="">
                            <h3 data-aos="fade-down"
                            data-aos-duration="1500"
                            data-aos-delay="50" 
                            style="font-size:35px; font-family:noto sans kr; display: inline-block">창작의 고통 없는</h3><br>
                            <h3 data-aos="fade-left"
                            data-aos-duration="1500"
                            data-aos-delay="50"
                            style="font-size:35px; font-family:noto sans kr; display: inline-block">인공지능 작가</h3><br>
                            <h3 data-aos="fade-left"
                            data-aos-duration="1500"
                            data-aos-delay="50"
                            style="color:#FF80AB;font-size:40px; font-family:noto sans kr; display: inline-block">ALPHA BOOK</h3><br><br>
                            <p class="subheading"
                            data-aos="fade-right"
                            data-aos-duration="2000"
                            data-aos-delay="350"
                            style="font-family:noto sans kr; "
                            >SNS, 수필, 일기, 블로그 포스팅 등<br/>
                            40초 만에 이미지로부터 한 편의 이야기를 만들어 보세요!</p>
                            <v-btn
                            class="mt-10"
                            data-aos="fade-up"
                            data-aos-duration="2000"
                            data-aos-delay="500"
                            rounded x-large style="background-color:#EB539E;padding-left:100px;padding-right:100px;" dark @click="openDialog(true)"> 바로 시작하기 </v-btn>
                        </div>
                        </v-flex>
                    </v-layout>
                </v-col>
                <v-col cols="5" class="mt-10">
                    <video width='450' height='300' poster='../../assets/poster.png' controls>
                        <source src='../../assets/mp4/UCC.mp4'>
                    </video>
                </v-col>
            </v-row>
        </section>

        <v-row style="height: 100px;background-color:white;"></v-row>

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
    </v-container>
</template>

<script>
import axios_common from '../../axios_common';
import { mapGetters } from 'vuex';
import Upload from "./ImageUploader.vue";

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
        data() {
            return {
                overlay: false,
                items: [
                    [
                    {
                        src: 'sample1.jpg'
                    }, {
                        src: 'sample2.jpg'
                    }, {
                        src: 'sample3.jpg'
                    }],
                    [
                    {
                        src: 'sample4.jpg'
                    }, {
                        src: 'sample5.jpg'
                    }, {
                        src: 'sample6.jpg'
                    }
                    ]
                ],
                storys : [],
                first_image : [],
                like_stories: [],
                stories_like_cnt: {},
                keyword:'',
                dialog: false, //true : Dialog열림, false : Dialog닫힘
            }
        },
        methods: {
            getImgUrl: function(path) {
                return require("@/assets/img/sampleImage/"+path);
            },
            like(id){
                if (!this.isAuthenticated) {
                    alert("로그인 해주세요.");
                }else {
                    axios_common.get(`/like/${id}`, this.requestHeader)
                    .then( response => {
                        const onLike = response.data.on_like

                        if(onLike){
                            this.like_stories.push(id);
                            this.stories_like_cnt[id] += 1;
                            // console.log(this.like_stories)
                        }else{
                            const idx = this.like_stories.indexOf(id);
                            if (idx > -1) this.like_stories.splice(idx, 1);
                            this.stories_like_cnt[id] -= 1;
                        }

                            
                    })
                    .catch(error => console.log(error))
                }
            },
            search(keyword){
                axios_common.post('/sub3/search/', {'keyword':keyword})
                .then(response => {
                    // console.log(response.data)
                    this.storys = response.data
                })
                .catch(error => console.log(error))
            },
            detail(story) {
                this.$router.push({ name: 'detail', params: {story: story}});
            },
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
        mounted() {
            // console.log(this.isAuthenticated)
            axios_common.get('/sub3/allstory/')
                .then(response => {
                    // console.log(response.data)
                    this.storys = response.data
                    for (let i=0;i<response.data.length;i++){
                        this.stories_like_cnt[response.data[i].id] = response.data[i].like_users.length;
                    }
                })
                .catch(error => console.log(error))
            axios_common.get('/sub3/likestories/', this.requestHeader)
                .then(response => {
                    
                    for (let i=0;i<response.data.length;i++){
                        this.like_stories.push(response.data[i].id)
                    }
                })
                .catch(error => console.log(error))
            },
        }
</script>
<style scoped>
.bg {
    padding: 30px;
    background: rgb(244, 242, 243);
    /* background-size: cover; */
    /* background-repeat:no-repeat; */
    /* background-position: center; */
}
.font1 {font-family: 'Do Hyeon', sans-serif;}
#font2 {font-family: 'Black Han Sans', sans-serif;}
</style>
