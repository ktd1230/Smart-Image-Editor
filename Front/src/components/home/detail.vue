<template>
    <v-app style="margin-left:80px;margin-right:80px;margin-top:50px;">
        <v-container fluid="fluid" class="contents grey lighten-5 pa-0 px-5">
            <v-row justify="center">
                <v-col cols="8">
                    <v-row justify="center"> 
                        <v-col :md=divide(images.length,i) xs="12" sm="12" class="pa-0" v-for="(image,i) in images" :key="i">
                            <v-img class="img" :src="`http://i02c104.p.ssafy.io:8000/media/${image}`"></v-img>
                        </v-col>
                    </v-row>
                </v-col>
                <v-col cols="4">
                    <v-card class="sampletext text-center">
                        <v-card-title>{{story.title || ''}}</v-card-title>
                        <v-card-text>
                            <div icon color="#C62828" style="outline:0">
                                <p>{{story.text}}</p>
                                <button data-html2canvas-ignore="true" class="text-primary" @click="makePDF"><v-icon color="primary">mdi-download</v-icon> pdf로 저장하기</button>
                            </div>
                        </v-card-text>
                    </v-card>
                </v-col>
            </v-row>
        </v-container>
    </v-app>
</template>

<script>
import html2canvas from 'html2canvas'
import jsPDF from 'jspdf'

    export default {
        props: ['story'],
         mounted () {
            // 주소가 정해지면 받는 처리 
            // axios.get('http://127.0.0.1:8000/', this.requestHeader)
            // .then(response => {
            //         console.log(response.data)
            //     }
            // )
            // .catch(error => console.log(error))
            // console.log(this.story)
            const imgs = this.story.image.split(',')
            for (let i=0;i<imgs.length;i++) {
                this.images.push(imgs[i].split(':')[1].replace("'",'').replace("'",'').replace("}",'').replace(' ',''))
            }
            // console.log(this.images)
         },
        data() {
            return {
                images: [
                    // {
                    //     src: 'https://cdn.vuetifyjs.com/images/carousel/squirrel.jpg',
                    // },{
                    //     src: 'https://cdn.vuetifyjs.com/images/carousel/sky.jpg',
                    // },
                    // {
                    //     src: 'https://cdn.vuetifyjs.com/images/carousel/bird.jpg',
                    // },
                    // {
                    //     src: 'https://cdn.vuetifyjs.com/images/carousel/planet.jpg',
                    // },{
                    //     src: 'https://cdn.vuetifyjs.com/images/carousel/planet.jpg',
                    // },
                ],
                text: {
                    title : '축복의 결혼식',
                    content: '신랑 신부가 결혼식 파티 앞에서 사진을 찍기 위해 포즈를 취하고 있네요.파란 셔츠를 입은 남자가 사다리에 서 있답니다.두 남자가 서 있는 걸 보니 두 사람은 이미 결혼 준비를 마친 것 같죠?'
                }
            }

        },
        methods:{
            divide(num,idx){
                //9개부터는 고려 안함
                var half = Math.floor(num/2)
               if(num>=3){
                   if(half > idx)
                   {
                       return 12/half
                   }
                   else{
                       console.log(num-half)
                       return 12/(num-half)
                   }
               }
               else {
                return 12/num  
               }
            },
            makePDF(){
                html2canvas(document.querySelector('.contents'), { logging: false, useCORS: true, proxy: '/etc/proxy_image', }).then(
                    (canvas) => {
                        var imgData = canvas.toDataURL('image/jpeg', 1.0);
                        // console.log('Report Image URL: '+imgData);
                        var doc = new jsPDF('p','mm');
                        var imgWidth = 210; // 이미지 가로 길이(mm) A4 기준
                        var pageHeight = imgWidth * 1.414;  // 출력 페이지 세로 길이 계산 A4 기준
                        var imgHeight = canvas.height * imgWidth / canvas.width;
                        var heightLeft = imgHeight;

                        var position = 0;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
                        
                        doc.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight);
                        heightLeft -= pageHeight;

                        while (heightLeft >= 20) {
                            position = heightLeft - imgHeight;
                            doc.addPage();
                            doc.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight);
                            heightLeft -= pageHeight;
                        }
                        
                        doc.save('sample-file.pdf');
                    
                })
            },
        },
    }
</script>
<style scoped>
.img{
    height: 300px;
}
.img:hover{
    height: 500px;
}
</style>
