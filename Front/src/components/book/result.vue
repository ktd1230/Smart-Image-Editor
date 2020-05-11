<template>
  <div>
    <v-container class="grey lighten-5 pa-10">
      <v-row>
        <!-- <v-col cols="12" v-for="(item,i) in response.image" :key="i">
            {{item}}
        </v-col> -->
        <v-col md="6" xs="12" sm="12" class="pa-0">
          <v-carousel
          height="600"
          >
            <v-carousel-item
              v-for="(image,i) in items.image"
              :key="i"
              :src="`http://13.124.246.175:8000/media/${image}`"
              reverse-transition="fade-transition"
              transition="fade-transition"
            ></v-carousel-item>
        </v-carousel>
        </v-col>
        <v-col md="6" xs="12" sm="12">
            <v-container
            fluid pa-0
            id="scroll-target"
            style="max-height: 580px"
            class="overflow-y-auto overflow-x-hidden pl-3"
            >
            <v-row>
                <v-col cols="12" v-for="(item,i) in items.text" :key="i">
                    <v-hover
                    v-slot:default="{ hover }"
                    >
                     <v-card class="sampletext" max-width = 95% v-if="selectedImage === -1">
                        <v-card-title>{{i+1}}번 샘플</v-card-title>

                        <v-card-text class="text--primary">
                        <div>{{item}}</div>
                        <v-fade-transition>
                        <v-overlay v-if="hover" absolute="absolute" color="#036358">
                        <v-btn @click="select(i)">선택하기</v-btn>
                        </v-overlay>
                        </v-fade-transition>
                        </v-card-text>
                    </v-card>

                    </v-hover>
                    <!-- <v-card class="sampletext" max-width = 95% v-if="selectedImage >= 0">
                        <v-card-title>{{selectedImage}}번 샘플</v-card-title>

                        <v-card-text class="text--primary">
                        <div>{{items[selectedImage].content}}</div>
                        <v-fade-transition>
                        <v-overlay v-if="hover" absolute="absolute" color="#036358">
                        <v-btn @click="test(i)">선택하기</v-btn>
                        </v-overlay>
                        </v-fade-transition>
                        </v-card-text>
                    </v-card> -->
                </v-col>
                <v-col cols="12" v-if="selectedImage >= 0">
                <v-card class="sampletext" max-width = 95%
                data-aos="fade-down"
                data-aos-duration="3000"
                data-aos-delay="150"
                >
                        <v-card-title>
                    <v-col cols="5">
                        <v-text-field 
                        single-line
                        :disabled="!enabled"
                        v-model="title"
                        >
                        <template v-slot:label >
                        <strong>{{selectedImage+1}}번</strong> 샘플
                         </template>    
                        </v-text-field>
                    </v-col>
                        </v-card-title>
                        <v-card-text class="text--primary">
                        <v-textarea
                        filled
                        auto-grow
                        rows="2"
                        row-height="20"
                        :value="items.text[selectedImage]"
                        :disabled="!enabled"
                        v-model="content"
                        ></v-textarea>
                        <v-col align="end">
                        <v-btn @click="back">이전으로</v-btn>
                        <v-btn color="#ff5989" dark @click="enabled=!enabled">수정하기</v-btn>
                        <v-btn color="#4CAF50" dark @click="save">저장하기</v-btn>
                        </v-col>
                        </v-card-text>
                    </v-card>
                </v-col>
                    


            </v-row>
           
            <!-- <v-btn  data-aos="fade-right"
                    data-aos-duration="1000"
                    data-aos-delay="150"
                    text>Lorem ipsum dolor sit amet, nemore sapientem ei qui, no pri indoctum prodesset,
                omnis quidam utroque nam ei. Ne utamur similique repudiandae eum.</v-btn>
                <h1  data-aos="fade-right"
                    data-aos-duration="2000"
                    data-aos-delay="1000">Lorem ipsum dolor sit amet, nemore sapientem ei qui, no pri indoctum prodesset,
                omnis quidam utroque nam ei. Ne utamur similique repudiandae eum.</h1>
                <h1  data-aos="fade-right"
                    data-aos-duration="3000"
                    data-aos-delay="2000">Lorem ipsum dolor sit amet, nemore sapientem ei qui, no pri indoctum prodesset,
                omnis quidam utroque nam ei. Ne utamur similique repudiandae eum.</h1>
                <h1>Lorem ipsum dolor sit amet, nemore sapientem ei qui, no pri indoctum prodesset,
                omnis quidam utroque nam ei. Ne utamur similique repudiandae eum.</h1>
                <h1>Lorem ipsum dolor sit amet, nemore sapientem ei qui, no pri indoctum prodesset,
                omnis quidam utroque nam ei. Ne utamur similique repudiandae eum.</h1>
                <h1>Lorem ipsum dolor sit amet, nemore sapientem ei qui, no pri indoctum prodesset,
                omnis quidam utroque nam ei. Ne utamur similique repudiandae eum.</h1>
                <h1>Lorem ipsum dolor sit amet, nemore sapientem ei qui, no pri indoctum prodesset,
                omnis quidam utroque nam ei. Ne utamur similique repudiandae eum.</h1> -->
            </v-container>
        </v-col>

      </v-row>
    </v-container>
  </div>
</template>


<script>
import router from "../../router"
import axios_common from '../../axios_common';
import { mapGetters } from 'vuex';

    export default {
        props: ['response'],
          computed: {
            ...mapGetters([
                'isAuthenticated',
                'requestHeader',
                'userId',
                'username'
            ])
        },
        mounted () {
            // console.log(this.$vuetify.breakpoint)
            // bus.$emit('end:spinner');
        },
        created(){
            // console.log(this.$route.response)
            // console.log("result창")
            console.log(this.response)
        },
        data() {
            return {
                overlay: false,
                selectedImage:-1,
                meta : this.$route.response,
                enabled: false,
                edit_content:false,
                title:"",
                content: "",
                // this.response.text,
                images : this.response.image,
                // images: [
                //     {
                //         src: 'https://cdn.vuetifyjs.com/images/carousel/squirrel.jpg'
                //     }, 
                //     {
                //         src: 'https://cdn.vuetifyjs.com/images/carousel/sky.jpg'
                //     }, 
                //     {
                //         src: 'https://cdn.vuetifyjs.com/images/carousel/bird.jpg'
                //     },
                //     {
                //         src: 'https://cdn.vuetifyjs.com/images/carousel/planet.jpg'
                //     }, 
                //     {
                //         src: 'https://cdn.vuetifyjs.com/images/cards/desert.jpg'
                //     }, 
                //     {
                //         src: 'https://picsum.photos/510/300?random'
                //     }
                // ],
                items: this.response
                // {
                //     content:'신랑 신부가 결혼식 파티 앞에서 사진을 찍기 위해 포즈를 취하고 있네요.파란 셔츠를 입은 남자가 사다리에 서 있답니다.두 남자가 서 있는 걸 보니 두 사람은 이미 결혼 준비를 마친 것 같죠?' 
                // },
                // {
                //         content:'신랑 신부가 결혼식 파티 앞에서 사진을 찍기 위해 포즈를 취하고 있네요.파란 셔츠를 입은 남자가 사다리에 서 있군요?두 남자가 서 있는 모습의 사진인데요.', 
                // },
                // {
                //         content:' 신랑 신부가 결혼식 파티 앞에서 사진을 찍기 위해 포즈를 취하고 있네요.파란 셔츠를 입은 남자가 사다리에 서 있답니다. 두 남자가 서 있는 이 모습을 보고요',
                // },
                // {
                //     content:' 신랑 신부가 결혼식 파티 앞에서 사진을 찍기 위해 포즈를 취하고 있자, 신랑 신부의 얼굴이 살짝 공개됐다.파란 셔츠를 입은 남자가 사다리에 서 있더니 한 손에 카메라를 들고 있다.두 남자가 서 있는 이 사진의 주인공은 다름 아닌 신부 김예림이다.', 
                // },
                // {
                //     content:' 신랑 신부가 결혼식 파티 앞에서 사진을 찍기 위해 포즈를 취하고 있자, 신랑 신부의 얼굴이 살짝 공개됐다.파란 셔츠를 입은 남자가 사다리에 서 있더니, 그 옆에 앉아 있던 여성이 손을 내밀며 “안녕히 가세요”라고 인사를 건넸다.두 남자가 서 있는 동안 두 사람은 서로 눈을 마주치며 대화를 나눴다.', 
                // },
                // {
                //     content: ' 신랑 신부가 결혼식 파티 앞에서 사진을 찍기 위해 포즈를 취하고 있자, 신랑 신부의 얼굴이 살짝 공개됐다.파란 셔츠를 입은 남자가 사다리에 서 있더니 손을 흔들며 인사를 건네고 있다.두 남자가 서 있는 모습은 두 사람이 결혼준비를 하고 있다는 것을 짐작케 한다.'
                // }
                
            }
        },
        methods: {
        select(num){
            this.selectedImage=num
            console.log(this.items.text[0])
            this.content=this.items.text[this.selectedImage]
        },back(){
            this.selectedImage=-1;
        },save(){
            axios_common.post('/sub3/mystory/', {img:this.images, title : this.title, content : this.content}, this.requestHeader)
                .then(response => {
                    console.log(response.data)
                    router.push('/mybook')
                })
                .catch(error => console.log(error))
            },
        },
    }
</script>
